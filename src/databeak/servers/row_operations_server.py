"""Standalone row operations server for DataBeak using FastMCP server composition."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Import session management from the main package  
from ..exceptions import ColumnNotFoundError, InvalidParameterError
from ..models import OperationType
from ..models.tool_responses import (
    CellValueResult,
    ColumnDataResult,
    DeleteRowResult,
    InsertRowResult,
    RowDataResult,
    SetCellResult,
    UpdateRowResult,
)
from ..utils.validators import convert_pandas_na_list
from .server_utils import get_session_data

if TYPE_CHECKING:
    from ..models.csv_session import CSVSession

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
            raise ValueError("Row index must be non-negative")
        return v


class RowInsertRequest(BaseModel):
    """Request parameters for row insertion operations."""

    model_config = ConfigDict(extra="forbid")

    row_index: int = Field(description="Index where to insert row (-1 to append at end)")
    data: dict[str, CellValue] | list[CellValue] | str = Field(
        description="Row data as dict, list, or JSON string"
    )

    @field_validator("data")
    @classmethod
    def parse_json_data(cls, v: dict[str, CellValue] | list[CellValue] | str) -> dict[str, CellValue] | list[CellValue]:
        """Parse JSON string data for Claude Code compatibility."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if not isinstance(parsed, (dict, list)):
                    raise ValueError("JSON string must parse to dict or list")
                return parsed
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}") from e
        return v


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
            raise ValueError("Row index must be non-negative")
        return v

    @field_validator("data")
    @classmethod
    def parse_json_data(cls, v: dict[str, CellValue] | str) -> dict[str, CellValue]:
        """Parse JSON string data for Claude Code compatibility."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if not isinstance(parsed, dict):
                    raise ValueError("JSON string must parse to dict for updates")
                return parsed
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}") from e
        return v


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
            raise ValueError("Row indices must be non-negative")
        return v


# ============================================================================
# ROW OPERATIONS LOGIC (Synchronous for computational operations)
# ============================================================================


def get_cell_value(
    session_id: str,
    row_index: int,
    column: str | int,
    ctx: Context | None = None,  # noqa: ARG001
) -> CellValueResult:
    """Get the value of a specific cell with precise coordinate targeting and comprehensive metadata.

    Essential tool for AI assistants to inspect individual cell values with full coordinate
    context. Part of the inspection workflow: get_data_summary → get_row_data → get_cell_value.

    Args:
        session_id: Session identifier for the active CSV data session
        row_index: Row index using 0-based indexing (0 = first row, N-1 = last row)
        column: Column targeting options:
                - String: Column name (e.g., "name", "email", "status") - preferred for clarity
                - Integer: Column index 0-based (0 = first column, N-1 = last column)
        ctx: FastMCP context

    Returns:
        Detailed cell information containing:
        - success: bool operation status
        - value: Actual cell value (None if null/NaN, preserves original type)
        - coordinates: {"row": index, "column": actual_column_name}
        - data_type: Column data type (int64, float64, object, datetime64, etc.)

    Examples:
        # Read by column name (recommended)
        get_cell_value("session123", 0, "name")    # → "John Doe"
        get_cell_value("session123", 5, "email")   # → "john@example.com" or None

        # Read by column index
        get_cell_value("session123", 2, 1)        # Third row, second column

        # Null value handling
        get_cell_value("session123", 1, "phone")  # → None if cell is empty/null

    Raises:
        ToolError: If session invalid, row/column out of bounds, or other errors
    """
    try:
        session, df = get_session_data(session_id)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            raise ToolError(f"Row index {row_index} out of range (0-{len(df) - 1})")

        # Handle column specification
        if isinstance(column, int):
            # Column index
            if column < 0 or column >= len(df.columns):
                raise ToolError(f"Column index {column} out of range (0-{len(df.columns) - 1})")
            column_name = df.columns[column]
        else:
            # Column name
            if column not in df.columns:
                raise ColumnNotFoundError(column, list(df.columns))
            column_name = column

        # Get the cell value
        value = df.iloc[row_index, df.columns.get_loc(column_name)]

        # Handle pandas/numpy types for JSON serialization
        if pd.isna(value):
            value = None
        elif hasattr(value, "item"):  # numpy scalar
            value = value.item()

        # Get column data type
        data_type = str(df[column_name].dtype)

        # Record operation for history
        session.record_operation(
            OperationType.DATA_INSPECTION,
            {
                "operation": "get_cell_value",
                "row_index": row_index,
                "column": column_name,
                "value_type": type(value).__name__,
            },
        )

        return CellValueResult(
            value=value,
            coordinates={"row": row_index, "column": column_name},
            data_type=data_type,
        )

    except (ColumnNotFoundError, ToolError):
        raise
    except Exception as e:
        logger.error(f"Error getting cell value: {e!s}")
        raise ToolError(f"Error getting cell value: {e!s}") from e


def set_cell_value(
    session_id: str,
    row_index: int,
    column: str | int,
    value: CellValue,
    ctx: Context | None = None,  # noqa: ARG001
) -> SetCellResult:
    """Set the value of a specific cell with precise coordinate targeting and null value support.

    This function provides pixel-perfect cell editing capabilities optimized for AI assistants
    working with tabular data. Supports both column names and indices for flexible targeting.

    Args:
        session_id: Session identifier for the active CSV data session
        row_index: Row index using 0-based indexing (0 = first row, N-1 = last row)
        column: Column targeting options:
                - String: Column name (e.g., "name", "email", "status")
                - Integer: Column index 0-based (0 = first column, N-1 = last column)
        value: New cell value with full type support:
               - Strings: "text", "email@example.com", ""
               - Numbers: 42, 3.14, -100
               - Booleans: true, false
               - Null: null/None for missing data
               - Automatically preserves type based on column context
        ctx: FastMCP context

    Returns:
        Detailed operation result containing:
        - success: bool operation status
        - coordinates: {"row": index, "column": actual_column_name}
        - old_value: Previous cell value (None if was null)
        - new_value: New cell value (None if setting to null)
        - data_type: Column data type information

    Examples:
        # Set by column name
        set_cell_value("session123", 0, "name", "Jane Smith")

        # Set by column index
        set_cell_value("session123", 2, 1, 25)

        # Set to null value
        set_cell_value("session123", 1, "email", None)

        # Update status fields
        set_cell_value("session123", 3, "status", "completed")

    Raises:
        ToolError: If session invalid, row/column out of bounds, or other errors
    """
    try:
        session, df = get_session_data(session_id)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            raise ToolError(f"Row index {row_index} out of range (0-{len(df) - 1})")

        # Handle column specification
        if isinstance(column, int):
            # Column index
            if column < 0 or column >= len(df.columns):
                raise ToolError(f"Column index {column} out of range (0-{len(df.columns) - 1})")
            column_name = df.columns[column]
        else:
            # Column name
            if column not in df.columns:
                raise ColumnNotFoundError(column, list(df.columns))
            column_name = column

        # Get the old value for tracking
        old_value = df.iloc[row_index, df.columns.get_loc(column_name)]
        if pd.isna(old_value):
            old_value = None
        elif hasattr(old_value, "item"):  # numpy scalar
            old_value = old_value.item()

        # Set the new value
        df.iloc[row_index, df.columns.get_loc(column_name)] = value

        # Get the new value for tracking (after pandas type conversion)
        new_value = df.iloc[row_index, df.columns.get_loc(column_name)]
        if pd.isna(new_value):
            new_value = None
        elif hasattr(new_value, "item"):  # numpy scalar
            new_value = new_value.item()

        # Get column data type
        data_type = str(df[column_name].dtype)

        # Record operation for history
        session.record_operation(
            OperationType.DATA_MODIFICATION,
            {
                "operation": "set_cell_value",
                "row_index": row_index,
                "column": column_name,
                "old_value": old_value,
                "new_value": new_value,
            },
        )

        return SetCellResult(
            coordinates={"row": row_index, "column": column_name},
            old_value=old_value,
            new_value=new_value,
            data_type=data_type,
        )

    except (ColumnNotFoundError, ToolError):
        raise
    except Exception as e:
        logger.error(f"Error setting cell value: {e!s}")
        raise ToolError(f"Error setting cell value: {e!s}") from e


def get_row_data(
    session_id: str,
    row_index: int,
    columns: list[str] | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> RowDataResult:
    """Get data from a specific row, optionally filtered by columns.

    Retrieves complete row data with optional column filtering for focused analysis.
    Essential for inspecting row context and validating data integrity.

    Args:
        session_id: Session identifier for the active CSV data session
        row_index: Row index using 0-based indexing (0 = first row, N-1 = last row)
        columns: Optional list of column names to include (None for all columns)
        ctx: FastMCP context

    Returns:
        RowDataResult containing:
        - success: bool operation status
        - session_id: Session identifier for continued operations
        - row_index: The row that was retrieved
        - data: Dictionary mapping column names to values
        - columns: List of column names included in the result

    Examples:
        # Get all data from first row
        get_row_data("session123", 0)

        # Get specific columns from second row
        get_row_data("session123", 1, ["name", "age"])

    Raises:
        ToolError: If session invalid, row out of bounds, or column not found
    """
    try:
        session, df = get_session_data(session_id)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            raise ToolError(f"Row index {row_index} out of range (0-{len(df) - 1})")

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

        # Record operation for history
        session.record_operation(
            OperationType.DATA_INSPECTION,
            {
                "operation": "get_row_data",
                "row_index": row_index,
                "columns_retrieved": len(selected_columns),
                "filtered": columns is not None,
            },
        )

        return RowDataResult(
            session_id=session_id,
            row_index=row_index,
            data=row_data,
            columns=selected_columns,
        )

    except (ColumnNotFoundError, ToolError):
        raise
    except Exception as e:
        logger.error(f"Error getting row data: {e!s}")
        raise ToolError(f"Error getting row data: {e!s}") from e


def get_column_data(
    session_id: str,
    column: str,
    start_row: int | None = None,
    end_row: int | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnDataResult:
    """Get data from a specific column, optionally sliced by row range.

    Retrieves column data with optional row range filtering for focused analysis.
    Useful for inspecting column patterns, distributions, and data quality.

    Args:
        session_id: Session identifier for the active CSV data session
        column: Column name to retrieve data from
        start_row: Starting row index (0-based, inclusive). None for beginning
        end_row: Ending row index (0-based, exclusive). None for end
        ctx: FastMCP context

    Returns:
        ColumnDataResult containing:
        - success: bool operation status
        - session_id: Session identifier for continued operations
        - column: Column name that was retrieved
        - values: List of column values in the specified range
        - total_values: Number of values returned
        - start_row: Starting row index used (None if from beginning)
        - end_row: Ending row index used (None if to end)

    Examples:
        # Get all values from "age" column
        get_column_data("session123", "age")

        # Get first 5 values from "name" column
        get_column_data("session123", "name", 0, 5)

        # Get values from row 10 onwards
        get_column_data("session123", "email", 10)

    Raises:
        ToolError: If session invalid, column not found, or invalid row range
    """
    try:
        session, df = get_session_data(session_id)

        # Validate column exists
        if column not in df.columns:
            raise ColumnNotFoundError(column, list(df.columns))

        # Validate and set row range
        if start_row is not None and start_row < 0:
            raise InvalidParameterError("start_row", start_row, "must be non-negative")
        if end_row is not None and end_row < 0:
            raise InvalidParameterError("end_row", end_row, "must be non-negative")
        if start_row is not None and start_row >= len(df):
            raise ToolError(f"start_row {start_row} out of range (0-{len(df) - 1})")
        if end_row is not None and end_row > len(df):
            raise ToolError(f"end_row {end_row} out of range (0-{len(df)})")
        if start_row is not None and end_row is not None and start_row >= end_row:
            raise InvalidParameterError("start_row", start_row, "must be less than end_row")

        # Apply row range slicing
        if start_row is None and end_row is None:
            column_data = df[column]
        elif start_row is None:
            column_data = df[column][:end_row]
        elif end_row is None:
            column_data = df[column][start_row:]
        else:
            column_data = df[column][start_row:end_row]

        # Convert to list and handle pandas/numpy types
        values = convert_pandas_na_list(column_data.tolist())

        # Record operation for history
        session.record_operation(
            OperationType.DATA_INSPECTION,
            {
                "operation": "get_column_data",
                "column": column,
                "values_retrieved": len(values),
                "start_row": start_row,
                "end_row": end_row,
            },
        )

        return ColumnDataResult(
            session_id=session_id,
            column=column,
            values=values,
            total_values=len(values),
            start_row=start_row,
            end_row=end_row,
        )

    except (ColumnNotFoundError, InvalidParameterError, ToolError):
        raise
    except Exception as e:
        logger.error(f"Error getting column data: {e!s}")
        raise ToolError(f"Error getting column data: {e!s}") from e


def insert_row(
    session_id: str,
    row_index: int,
    data: RowData | str,  # Accept string for Claude Code compatibility
    ctx: Context | None = None,  # noqa: ARG001
) -> InsertRowResult:
    """Insert a new row at the specified index with comprehensive null value and JSON string support.

    This function is optimized for AI assistants and supports multiple data input formats including
    automatic JSON string parsing for Claude Code compatibility.

    Args:
        session_id: Session identifier for the active CSV data session
        row_index: Index where to insert the row (0-based indexing). Use -1 to append at end
        data: Row data in multiple formats:
              - Dict: {"column_name": value, ...} with all or partial columns
              - List: [value1, value2, ...] matching column order exactly
              - JSON string: Automatically parsed from Claude Code serialization

              All formats support null/None values:
              - JSON null → Python None → pandas NaN
              - Missing dict keys → filled with None
              - Explicit None values → preserved as pandas NaN
        ctx: FastMCP context

    Returns:
        Comprehensive operation result containing:
        - success: bool indicating operation status
        - operation: "insert_row" for tracking
        - row_index: Actual insertion index used
        - rows_before/rows_after: Data size changes
        - data_inserted: The actual row data that was inserted
        - columns: Current column list
        - session_id: Session identifier for continued operations

    Examples:
        # Standard dictionary insertion
        insert_row("session123", 1, {"name": "Alice", "age": 28, "city": "Boston"})

        # List insertion (append)
        insert_row("session123", -1, ["David", 40, "Miami"])

        # Null value support
        insert_row("session123", 0, {"name": "John", "age": None, "city": "NYC"})

        # Claude Code JSON string (automatically handled)
        insert_row("session123", 1, '{"name": "Alice", "email": null, "status": "active"}')

    Raises:
        ToolError: If session invalid, row index out of range, or data format errors
    """
    try:
        # Handle Claude Code's JSON string serialization
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                raise ToolError(f"Invalid JSON string in data parameter: {e}") from e

        session, df = get_session_data(session_id)
        rows_before = len(df)

        # Handle special case: append at end
        if row_index == -1:
            row_index = len(df)

        # Validate row index for insertion (0 to N is valid for insertion)
        if row_index < 0 or row_index > len(df):
            raise ToolError(f"Row index {row_index} out of range for insertion (0-{len(df)})")

        # Process data based on type
        if isinstance(data, dict):
            # Dictionary format - fill missing columns with None
            row_data = {}
            for col in df.columns:
                row_data[col] = data.get(col, None)
        elif isinstance(data, list):
            # List format - must match column count
            if len(data) != len(df.columns):
                raise ToolError(
                    f"List data length ({len(data)}) must match column count ({len(df.columns)})"
                )
            row_data = dict(zip(df.columns, data))
        else:
            raise ToolError(f"Unsupported data type: {type(data)}. Use dict, list, or JSON string")

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
        data_inserted = {}
        for key, value in row_data.items():
            if pd.isna(value):
                data_inserted[key] = None
            elif hasattr(value, "item"):  # numpy scalar
                data_inserted[key] = value.item()
            else:
                data_inserted[key] = value

        # Record operation for history
        session.record_operation(
            OperationType.DATA_MODIFICATION,
            {
                "operation": "insert_row",
                "row_index": row_index,
                "rows_added": 1,
                "data_type": "dict" if isinstance(data, dict) else "list",
            },
        )

        return InsertRowResult(
            row_index=row_index,
            rows_before=rows_before,
            rows_after=len(df_new),
            data_inserted=data_inserted,
            columns=list(df_new.columns),
            session_id=session_id,
        )

    except ToolError:
        raise
    except Exception as e:
        logger.error(f"Error inserting row: {e!s}")
        raise ToolError(f"Error inserting row: {e!s}") from e


def delete_row(
    session_id: str,
    row_index: int,
    ctx: Context | None = None,  # noqa: ARG001
) -> DeleteRowResult:
    """Delete a row at the specified index.

    Removes a row from the dataset with comprehensive tracking and validation.
    Provides detailed information about the deleted data for undo operations.

    Args:
        session_id: Session identifier for the active CSV data session
        row_index: Row index to delete using 0-based indexing (0 = first row, N-1 = last row)
        ctx: FastMCP context

    Returns:
        DeleteRowResult containing:
        - success: bool operation status
        - session_id: Session identifier for continued operations
        - operation: "delete_row" for tracking
        - row_index: The row that was deleted
        - rows_before/rows_after: Data size changes
        - deleted_data: Dictionary containing the deleted row data

    Examples:
        # Delete second row (index 1)
        delete_row("session123", 1)

        # Delete first row (index 0)
        delete_row("session123", 0)

        # Delete last row
        delete_row("session123", len(df) - 1)

    Raises:
        ToolError: If session invalid or row index out of bounds
    """
    try:
        session, df = get_session_data(session_id)
        rows_before = len(df)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            raise ToolError(f"Row index {row_index} out of range (0-{len(df) - 1})")

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

        # Record operation for history
        session.record_operation(
            OperationType.DATA_MODIFICATION,
            {
                "operation": "delete_row",
                "row_index": row_index,
                "rows_removed": 1,
                "deleted_columns": list(deleted_data.keys()),
            },
        )

        return DeleteRowResult(
            session_id=session_id,
            row_index=row_index,
            rows_before=rows_before,
            rows_after=len(df_new),
            deleted_data=deleted_data,
        )

    except ToolError:
        raise
    except Exception as e:
        logger.error(f"Error deleting row: {e!s}")
        raise ToolError(f"Error deleting row: {e!s}") from e


def update_row(
    session_id: str,
    row_index: int,
    data: dict[str, CellValue] | str,
    ctx: Context | None = None,  # noqa: ARG001
) -> UpdateRowResult:
    """Update specific columns in a row with comprehensive null value and Claude Code JSON string support.

    Provides selective column updates within a single row, supporting partial updates and
    automatic JSON string parsing. Optimized for AI assistants with detailed change tracking.

    Args:
        session_id: Session identifier for the active CSV data session
        row_index: Row index using 0-based indexing (0 = first row, N-1 = last row)
        data: Row update data in multiple formats:
              - Dict: {"column_name": new_value, ...} for partial column updates
              - JSON string: Automatically parsed from Claude Code serialization

              All formats support null/None values:
              - JSON null → Python None → pandas NaN
              - Explicit None values → preserved as pandas NaN
        ctx: FastMCP context

    Returns:
        Detailed update result containing:
        - success: bool operation status
        - operation: "update_row" for tracking
        - row_index: The row that was updated
        - columns_updated: List of column names that were changed
        - old_values: Previous values for changed columns (None if was null)
        - new_values: New values for changed columns (None if set to null)
        - changes_made: Number of columns updated

    Examples:
        # Standard dictionary update
        update_row("session123", 0, {"age": 31, "city": "Boston"})

        # Set values to null
        update_row("session123", 1, {"phone": None, "email": None})

        # Partial update (only specified columns changed)
        update_row("session123", 2, {"status": "completed"})

    Raises:
        ToolError: If session invalid, row out of bounds, or column not found
    """
    try:
        # Handle Claude Code's JSON string serialization
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                raise ToolError(f"Invalid JSON string in data parameter: {e}") from e

        if not isinstance(data, dict):
            raise ToolError("Update data must be a dictionary or JSON string")

        session, df = get_session_data(session_id)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            raise ToolError(f"Row index {row_index} out of range (0-{len(df) - 1})")

        # Validate all columns exist
        missing_columns = [col for col in data.keys() if col not in df.columns]
        if missing_columns:
            raise ColumnNotFoundError(missing_columns[0], list(df.columns))

        # Track changes
        columns_updated = []
        old_values = {}
        new_values = {}

        # Update each column
        for column, new_value in data.items():
            # Get old value
            old_value = df.iloc[row_index, df.columns.get_loc(column)]
            if pd.isna(old_value):
                old_value = None
            elif hasattr(old_value, "item"):  # numpy scalar
                old_value = old_value.item()

            # Set new value
            df.iloc[row_index, df.columns.get_loc(column)] = new_value

            # Get new value (after pandas type conversion)
            updated_value = df.iloc[row_index, df.columns.get_loc(column)]
            if pd.isna(updated_value):
                updated_value = None
            elif hasattr(updated_value, "item"):  # numpy scalar
                updated_value = updated_value.item()

            # Track the change
            columns_updated.append(column)
            old_values[column] = old_value
            new_values[column] = updated_value

        # Record operation for history
        session.record_operation(
            OperationType.DATA_MODIFICATION,
            {
                "operation": "update_row",
                "row_index": row_index,
                "columns_updated": len(columns_updated),
                "columns": columns_updated,
            },
        )

        return UpdateRowResult(
            row_index=row_index,
            columns_updated=columns_updated,
            old_values=old_values,
            new_values=new_values,
            changes_made=len(columns_updated),
        )

    except (ColumnNotFoundError, ToolError):
        raise
    except Exception as e:
        logger.error(f"Error updating row: {e!s}")
        raise ToolError(f"Error updating row: {e!s}") from e


# ============================================================================
# FASTMCP SERVER SETUP
# ============================================================================

# Create row operations server
row_operations_server = FastMCP(
    "DataBeak-RowOperations", instructions="Row operations server for DataBeak"
)

# Register functions directly as MCP tools (no wrapper functions needed)
row_operations_server.tool(name="get_cell_value")(get_cell_value)
row_operations_server.tool(name="set_cell_value")(set_cell_value)
row_operations_server.tool(name="get_row_data")(get_row_data)
row_operations_server.tool(name="get_column_data")(get_column_data)
row_operations_server.tool(name="insert_row")(insert_row)
row_operations_server.tool(name="delete_row")(delete_row)
row_operations_server.tool(name="update_row")(update_row)