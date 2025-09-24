"""Standalone row operations server for DataBeak using FastMCP server composition."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

import pandas as pd
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from databeak.core.session import get_session_data

# Import session management from the main package
from databeak.exceptions import ColumnNotFoundError, InvalidParameterError
from databeak.models.tool_responses import (
    CellValueResult,
    ColumnDataResult,
    DeleteRowResult,
    InsertRowResult,
    RowDataResult,
    SetCellResult,
    UpdateRowResult,
)
from databeak.utils.pydantic_validators import (
    parse_json_string_to_dict,
    parse_json_string_to_dict_or_list,
)
from databeak.utils.validators import convert_pandas_na_list

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Type aliases for better type safety
CellValue = str | int | float | bool | None
RowData = dict[str, CellValue] | list[CellValue]

# ============================================================================
# PYDANTIC MODELS FOR REQUEST PARAMETERS
# ============================================================================


class CellCoordinates(BaseModel):
    """Cell coordinate specification for precise targeting."""

    model_config = ConfigDict(extra="forbid")

    row_index: int = Field(ge=0, description="Row index using 0-based indexing")
    column: str | int = Field(description="Column name (str) or column index (int)")

    @field_validator("row_index")
    @classmethod
    def validate_row_index(cls, v: int) -> int:
        """Validate row index is non-negative."""
        if v < 0:
            msg = "Row index must be non-negative"
            raise ValueError(msg)
        return v


class RowInsertRequest(BaseModel):
    """Request parameters for row insertion operations."""

    model_config = ConfigDict(extra="forbid")

    row_index: int = Field(description="Index where to insert row (-1 to append at end)")
    data: dict[str, CellValue] | list[CellValue] | str = Field(
        description="Row data as dict, list, or JSON string",
    )

    @field_validator("data")
    @classmethod
    def parse_json_data(
        cls,
        v: dict[str, CellValue] | list[CellValue] | str,
    ) -> dict[str, CellValue] | list[CellValue]:
        """Parse JSON string data for Claude Code compatibility."""
        return parse_json_string_to_dict_or_list(v)


class RowUpdateRequest(BaseModel):
    """Request parameters for row update operations."""

    model_config = ConfigDict(extra="forbid")

    row_index: int = Field(ge=0, description="Row index to update (0-based)")
    data: dict[str, CellValue] | str = Field(description="Column updates as dict or JSON string")

    @field_validator("row_index")
    @classmethod
    def validate_row_index(cls, v: int) -> int:
        """Validate row index is non-negative."""
        if v < 0:
            msg = "Row index must be non-negative"
            raise ValueError(msg)
        return v

    @field_validator("data")
    @classmethod
    def parse_json_data(cls, v: dict[str, CellValue] | str) -> dict[str, CellValue]:
        """Parse JSON string data for Claude Code compatibility."""
        return parse_json_string_to_dict(v)


class ColumnDataRequest(BaseModel):
    """Request parameters for column data retrieval."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column name")
    start_row: int | None = Field(None, ge=0, description="Starting row index (inclusive)")
    end_row: int | None = Field(None, ge=0, description="Ending row index (exclusive)")

    @field_validator("start_row", "end_row")
    @classmethod
    def validate_row_indices(cls, v: int | None) -> int | None:
        """Validate row indices are non-negative."""
        if v is not None and v < 0:
            msg = "Row indices must be non-negative"
            raise ValueError(msg)
        return v


# ============================================================================
# ROW OPERATIONS LOGIC (Synchronous for computational operations)
# ============================================================================


# Implementation: validates row/column bounds, handles column name or index
# Preserves original data types, converts pandas NaN to None for JSON serialization
# Records operation in session history for audit trail
def get_cell_value(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    row_index: Annotated[int, Field(description="Row index (0-based) to retrieve cell from")],
    column: Annotated[
        str | int,
        Field(description="Column name or column index (0-based) to retrieve"),
    ],
) -> CellValueResult:
    """Get value of specific cell with coordinate targeting.

    Supports column name or index targeting. Returns value with coordinates and data type
    information.
    """
    try:
        session_id = ctx.session_id
        _session, df = get_session_data(session_id)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            msg = f"Row index {row_index} out of range (0-{len(df) - 1})"
            raise ToolError(msg)

        # Handle column specification
        if isinstance(column, int):
            # Column index
            if column < 0 or column >= len(df.columns):
                msg = f"Column index {column} out of range (0-{len(df.columns) - 1})"
                raise ToolError(msg)
            column_name = df.columns[column]
        else:
            # Column name
            if column not in df.columns:
                raise ColumnNotFoundError(column, list(df.columns))
            column_name = column

        # Get the cell value
        value = df.iloc[row_index, df.columns.get_loc(column_name)]  # type: ignore[index]

        # Handle pandas/numpy types for JSON serialization
        if pd.isna(value):
            value = None
        elif hasattr(value, "item"):  # numpy scalar
            value = value.item()  # type: ignore[assignment]

        # Get column data type
        data_type = str(df[column_name].dtype)

        # No longer recording operations (simplified MCP architecture)

        return CellValueResult(
            value=value,  # type: ignore[arg-type]
            coordinates={"row": row_index, "column": column_name},
            data_type=data_type,
        )

    except (ColumnNotFoundError, ToolError):
        raise
    except (ValueError, TypeError, KeyError, IndexError) as e:
        logger.exception("Error getting cell value: %s", str(e))
        msg = f"Error getting cell value: {e!s}"
        raise ToolError(msg) from e


# Implementation: validates coordinates, tracks old/new values for audit
# Supports column name or index, handles type conversion and null values
# Records operation in session history and auto-saves if enabled
def set_cell_value(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    row_index: Annotated[int, Field(description="Row index (0-based) to update cell in")],
    column: Annotated[
        str | int,
        Field(description="Column name or column index (0-based) to update"),
    ],
    value: Annotated[
        CellValue,
        Field(description="New value to set in the cell (str, int, float, bool, or None)"),
    ],
) -> SetCellResult:
    """Set value of specific cell with coordinate targeting.

    Supports column name or index, tracks old and new values. Returns operation result with
    coordinates and data type.
    """
    try:
        session_id = ctx.session_id
        _session, df = get_session_data(session_id)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            msg = f"Row index {row_index} out of range (0-{len(df) - 1})"
            raise ToolError(msg)

        # Handle column specification
        if isinstance(column, int):
            # Column index
            if column < 0 or column >= len(df.columns):
                msg = f"Column index {column} out of range (0-{len(df.columns) - 1})"
                raise ToolError(msg)
            column_name = df.columns[column]
        else:
            # Column name
            if column not in df.columns:
                raise ColumnNotFoundError(column, list(df.columns))
            column_name = column

        # Get the old value for tracking
        old_value = df.iloc[row_index, df.columns.get_loc(column_name)]  # type: ignore[index]
        if pd.isna(old_value):
            old_value = None
        elif hasattr(old_value, "item"):  # numpy scalar
            old_value = old_value.item()  # type: ignore[assignment]

        # Set the new value with explicit type conversion to avoid dtype compatibility warnings
        col_idx = df.columns.get_loc(column_name)
        current_dtype = df[column_name].dtype  # Access dtype through column name instead

        # Convert value to match column dtype if possible
        converted_value: CellValue
        try:
            if pd.api.types.is_numeric_dtype(current_dtype) and isinstance(value, str):
                numeric_result = pd.to_numeric(value, errors="coerce")
                # Convert pandas numeric to Python type for CellValue compatibility
                if pd.isna(numeric_result):
                    converted_value = None
                else:
                    converted_value = (
                        float(numeric_result)
                        if isinstance(numeric_result, (int, float))
                        else numeric_result.item()
                    )
            else:
                converted_value = value
        except (ValueError, TypeError):
            converted_value = value

        df.iloc[row_index, col_idx] = converted_value  # type: ignore[index]

        # Get the new value for tracking (after pandas type conversion)
        new_value = df.iloc[row_index, df.columns.get_loc(column_name)]  # type: ignore[index]
        if pd.isna(new_value):
            new_value = None
        elif hasattr(new_value, "item"):  # numpy scalar
            new_value = new_value.item()  # type: ignore[assignment]

        # Get column data type
        data_type = str(df[column_name].dtype)

        # No longer recording operations (simplified MCP architecture)

        return SetCellResult(
            coordinates={"row": row_index, "column": column_name},
            old_value=old_value,  # type: ignore[arg-type]
            new_value=new_value,  # type: ignore[arg-type]
            data_type=data_type,
        )

    except (ColumnNotFoundError, ToolError):
        raise
    except (ValueError, TypeError, KeyError, IndexError) as e:
        logger.exception("Error setting cell value: %s", str(e))
        msg = f"Error setting cell value: {e!s}"
        raise ToolError(msg) from e


# Implementation: validates row bounds, optional column filtering
# Converts pandas types to JSON-serializable values, handles NaN values
# Records operation in session history for audit trail
def get_row_data(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    row_index: Annotated[int, Field(description="Row index (0-based) to retrieve data from")],
    columns: Annotated[
        list[str] | None,
        Field(description="Optional list of column names to retrieve (all columns if None)"),
    ] = None,
) -> RowDataResult:
    """Get data from specific row with optional column filtering.

    Returns complete row data or filtered by column list. Converts pandas types for JSON
    serialization.
    """
    try:
        session_id = ctx.session_id
        _session, df = get_session_data(session_id)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            msg = f"Row index {row_index} out of range (0-{len(df) - 1})"
            raise ToolError(msg)

        # Handle column filtering
        if columns is None:
            selected_columns = list(df.columns)
            row_data = df.iloc[row_index].to_dict()
        else:
            # Validate all columns exist
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                raise ColumnNotFoundError(missing_columns[0], list(df.columns))

            selected_columns = columns
            row_data = df.iloc[row_index][columns].to_dict()

        # Handle pandas/numpy types for JSON serialization
        for key, value in row_data.items():
            if pd.isna(value):
                row_data[key] = None
            elif hasattr(value, "item"):  # numpy scalar
                row_data[key] = value.item()

        # No longer recording operations (simplified MCP architecture)

        return RowDataResult(
            row_index=row_index,
            data=row_data,
            columns=selected_columns,
        )

    except (ColumnNotFoundError, ToolError):
        raise
    except Exception as e:
        logger.exception("Error getting row data: %s", str(e))
        msg = f"Error getting row data: {e!s}"
        raise ToolError(msg) from e


# Implementation: validates column exists and row range bounds
# Supports optional row slicing with start_row (inclusive) and end_row (exclusive)
# Converts pandas types to JSON-serializable values, handles NaN conversion
def get_column_data(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    column: Annotated[str, Field(description="Column name to retrieve data from")],
    start_row: Annotated[
        int | None,
        Field(description="Starting row index (inclusive, 0-based) for data slice"),
    ] = None,
    end_row: Annotated[
        int | None,
        Field(description="Ending row index (exclusive, 0-based) for data slice"),
    ] = None,
) -> ColumnDataResult:
    """Get data from specific column with optional row range slicing.

    Supports row range filtering for focused analysis. Returns column values with range metadata.
    """
    try:
        session_id = ctx.session_id
        _session, df = get_session_data(session_id)

        # Validate column exists
        if column not in df.columns:
            raise ColumnNotFoundError(column, list(df.columns))

        # Validate and set row range
        if start_row is not None and start_row < 0:
            msg = "start_row"
            raise InvalidParameterError(msg, start_row, "must be non-negative")
        if end_row is not None and end_row < 0:
            msg = "end_row"
            raise InvalidParameterError(msg, end_row, "must be non-negative")
        if start_row is not None and start_row >= len(df):
            msg = f"start_row {start_row} out of range (0-{len(df) - 1})"
            raise ToolError(msg)
        if end_row is not None and end_row > len(df):
            msg = f"end_row {end_row} out of range (0-{len(df)})"
            raise ToolError(msg)
        if start_row is not None and end_row is not None and start_row >= end_row:
            msg = "start_row"
            raise InvalidParameterError(msg, start_row, "must be less than end_row")

        # Apply row range slicing
        if start_row is None and end_row is None:
            column_data = df[column]
            start_row = 0
            end_row = len(df) - 1
        elif start_row is None:
            column_data = df[column][:end_row]
            start_row = 0
        elif end_row is None:
            column_data = df[column][start_row:]
            end_row = len(df) - 1
        else:
            column_data = df[column][start_row:end_row]

        # Convert to list and handle pandas/numpy types
        values = convert_pandas_na_list(column_data.tolist())

        # No longer recording operations (simplified MCP architecture)

        return ColumnDataResult(
            column=column,
            values=values,
            total_values=len(values),
            start_row=start_row,
            end_row=end_row,
        )

    except (ColumnNotFoundError, InvalidParameterError, ToolError):
        raise
    except Exception as e:
        logger.exception("Error getting column data: %s", str(e))
        msg = f"Error getting column data: {e!s}"
        raise ToolError(msg) from e


# Implementation: supports dict, list, and JSON string data formats
# Validates row_index bounds (-1 for append), auto-parses JSON strings from Claude Code
# Handles null values, missing dict keys filled with None, records operation history
def insert_row(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    row_index: Annotated[
        int,
        Field(description="Index to insert row at (0-based, -1 to append at end)"),
    ],
    data: Annotated[
        RowData | str,
        Field(description="Row data as dict, list, or JSON string"),
    ],  # Accept string for Claude Code compatibility
) -> InsertRowResult:
    """Insert new row at specified index with multiple data formats.

    Supports dict, list, and JSON string input with null value handling. Returns insertion result
    with before/after statistics.
    """
    try:
        # Handle Claude Code's JSON string serialization
        if isinstance(data, str):
            try:
                data = parse_json_string_to_dict(data)
            except ValueError as e:
                msg = f"Invalid JSON string in data parameter: {e}"
                raise ToolError(msg) from e

        session_id = ctx.session_id
        session, df = get_session_data(session_id)
        rows_before = len(df)

        # Handle special case: append at end
        if row_index == -1:
            row_index = len(df)

        # Validate row index for insertion (0 to N is valid for insertion)
        if row_index < 0 or row_index > len(df):
            msg = f"Row index {row_index} out of range for insertion (0-{len(df)})"
            raise ToolError(msg)

        # Process data based on type
        if isinstance(data, dict):
            # Dictionary format - fill missing columns with None
            row_data = {}
            for col in df.columns:
                row_data[col] = data.get(col, None)
        elif isinstance(data, list):
            # List format - must match column count
            try:
                row_data = dict(zip(df.columns, data, strict=True))
            except ValueError as e:
                msg = f"List data length ({len(data)}) must match column count ({len(df.columns)})"
                raise ToolError(
                    msg,
                ) from e
        else:
            msg = f"Unsupported data type: {type(data)}. Use dict, list, or JSON string"
            raise ToolError(msg)

        # Create new row as DataFrame
        new_row = pd.DataFrame([row_data])

        # Insert the row
        if row_index == 0:
            # Insert at beginning
            df_new = pd.concat([new_row, df], ignore_index=True)
        elif row_index >= len(df):
            # Append at end
            df_new = pd.concat([df, new_row], ignore_index=True)
        else:
            # Insert in middle
            df_before = df.iloc[:row_index]
            df_after = df.iloc[row_index:]
            df_new = pd.concat([df_before, new_row, df_after], ignore_index=True)

        # Update session data
        session.df = df_new

        # Prepare inserted data for response (handle pandas types)
        data_inserted: dict[str, CellValue] = {}
        for key, value in row_data.items():
            if pd.isna(value):
                data_inserted[key] = None
            elif hasattr(value, "item"):  # numpy scalar
                data_inserted[key] = value.item()
            else:
                data_inserted[key] = value

        # No longer recording operations (simplified MCP architecture)

        return InsertRowResult(
            row_index=row_index,
            rows_before=rows_before,
            rows_after=len(df_new),
            data_inserted=data_inserted,
            columns=list(df_new.columns),
        )

    except ToolError:
        raise
    except Exception as e:
        logger.exception("Error inserting row: %s", str(e))
        msg = f"Error inserting row: {e!s}"
        raise ToolError(msg) from e


# Implementation: validates row_index bounds, captures deleted data for undo
# Updates DataFrame indexes after deletion, records operation in session history
# Provides comprehensive tracking with before/after row counts
def delete_row(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    row_index: Annotated[int, Field(description="Row index (0-based) to delete")],
) -> DeleteRowResult:
    """Delete row at specified index with comprehensive tracking.

    Captures deleted data for undo operations. Returns operation result with before/after
    statistics.
    """
    try:
        session_id = ctx.session_id
        session, df = get_session_data(session_id)
        rows_before = len(df)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            msg = f"Row index {row_index} out of range (0-{len(df) - 1})"
            raise ToolError(msg)

        # Get the data that will be deleted for tracking
        deleted_data = df.iloc[row_index].to_dict()

        # Handle pandas/numpy types for JSON serialization
        for key, value in deleted_data.items():
            if pd.isna(value):
                deleted_data[key] = None
            elif hasattr(value, "item"):  # numpy scalar
                deleted_data[key] = value.item()

        # Delete the row
        df_new = df.drop(df.index[row_index]).reset_index(drop=True)

        # Update session data
        session.df = df_new

        # No longer recording operations (simplified MCP architecture)

        return DeleteRowResult(
            row_index=row_index,
            rows_before=rows_before,
            rows_after=len(df_new),
            deleted_data=deleted_data,
        )

    except ToolError:
        raise
    except Exception as e:
        logger.exception("Error deleting row: %s", str(e))
        msg = f"Error deleting row: {e!s}"
        raise ToolError(msg) from e


# Implementation: validates row bounds, supports dict and JSON string data
# Selective column updates with change tracking (old/new values)
# Auto-parses JSON strings from Claude Code, records operation in session history
def update_row(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    row_index: Annotated[int, Field(description="Row index (0-based) to update")],
    data: Annotated[
        dict[str, CellValue] | str,
        Field(description="Column updates as dict mapping column names to values, or JSON string"),
    ],
) -> UpdateRowResult:
    """Update specific columns in row with selective updates.

    Supports partial column updates with change tracking. Returns old/new values for updated
    columns.
    """
    try:
        # Handle Claude Code's JSON string serialization
        if isinstance(data, str):
            try:
                data = parse_json_string_to_dict(data)
            except ValueError as e:
                msg = f"Invalid JSON string in data parameter: {e}"
                raise ToolError(msg) from e

        if not isinstance(data, dict):
            msg = "Update data must be a dictionary or JSON string"
            raise ToolError(msg)

        session_id = ctx.session_id
        _session, df = get_session_data(session_id)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            msg = f"Row index {row_index} out of range (0-{len(df) - 1})"
            raise ToolError(msg)

        # Validate all columns exist
        missing_columns = [col for col in data if col not in df.columns]
        if missing_columns:
            raise ColumnNotFoundError(missing_columns[0], list(df.columns))

        # Track changes
        columns_updated = []
        old_values = {}
        new_values = {}

        # Update each column
        for column, new_value in data.items():
            # Get old value
            old_value = df.iloc[row_index, df.columns.get_loc(column)]  # type: ignore[index]
            if pd.isna(old_value):
                old_value = None
            elif hasattr(old_value, "item"):  # numpy scalar
                old_value = old_value.item()  # type: ignore[assignment]

            # Set new value
            df.iloc[row_index, df.columns.get_loc(column)] = new_value  # type: ignore[index]

            # Get new value (after pandas type conversion)
            updated_value = df.iloc[row_index, df.columns.get_loc(column)]  # type: ignore[index]
            if pd.isna(updated_value):
                updated_value = None
            elif hasattr(updated_value, "item"):  # numpy scalar
                updated_value = updated_value.item()  # type: ignore[assignment]

            # Track the change
            columns_updated.append(column)
            old_values[column] = old_value
            new_values[column] = updated_value

        # No longer recording operations (simplified MCP architecture)

        return UpdateRowResult(
            row_index=row_index,
            columns_updated=columns_updated,
            old_values=old_values,  # type: ignore[arg-type]
            new_values=new_values,  # type: ignore[arg-type]
            changes_made=len(columns_updated),
        )

    except (ColumnNotFoundError, ToolError):
        raise
    except Exception as e:
        logger.exception("Error updating row: %s", str(e))
        msg = f"Error updating row: {e!s}"
        raise ToolError(msg) from e


# ============================================================================
# FASTMCP SERVER SETUP
# ============================================================================

# Create row operations server
row_operations_server = FastMCP(
    "DataBeak-RowOperations",
    instructions="Row operations server for DataBeak",
)

# Register functions directly as MCP tools (no wrapper functions needed)
row_operations_server.tool(name="get_cell_value")(get_cell_value)
row_operations_server.tool(name="set_cell_value")(set_cell_value)
row_operations_server.tool(name="get_row_data")(get_row_data)
row_operations_server.tool(name="get_column_data")(get_column_data)
row_operations_server.tool(name="insert_row")(insert_row)
row_operations_server.tool(name="delete_row")(delete_row)
row_operations_server.tool(name="update_row")(update_row)
