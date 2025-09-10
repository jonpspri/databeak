"""FastMCP server for column-level data operations.

This server provides column selection, renaming, addition, removal, and type conversion.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import pandas as pd
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from ..exceptions import (
    ColumnNotFoundError,
    InvalidParameterError,
    NoDataLoadedError,
    SessionNotFoundError,
)
from ..models import OperationType, get_session_manager
from ..models.tool_responses import (
    ColumnOperationResult,
    RenameColumnsResult,
    SelectColumnsResult,
)

logger = logging.getLogger(__name__)

# Type aliases
CellValue = str | int | float | bool | None

# =============================================================================
# PYDANTIC MODELS FOR REQUEST PARAMETERS
# =============================================================================


class ColumnMapping(BaseModel):
    """Column rename mapping."""

    old_name: str = Field(description="Current column name")
    new_name: str = Field(description="New column name")


class ColumnFormula(BaseModel):
    """Formula specification for computed columns."""

    expression: str = Field(description="Python expression for computing values")
    columns: list[str] = Field(default_factory=list, description="Columns referenced in expression")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_session_data(session_id: str) -> tuple[Any, pd.DataFrame]:
    """Get session and DataFrame, raising appropriate exceptions if not found."""
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if not session:
        raise SessionNotFoundError(session_id)
    if not session.data_session.has_data():
        raise NoDataLoadedError(session_id)

    df = session.data_session.df
    if df is None:  # Type guard since has_data() was checked
        raise NoDataLoadedError(session_id)
    return session, df


# =============================================================================
# TOOL DEFINITIONS (Direct implementations)
# =============================================================================


async def select_columns(
    session_id: str,
    columns: list[str],
    ctx: Context | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    """Select specific columns from the dataframe, removing all others.

    Args:
        session_id: Session identifier
        columns: List of column names to keep
        ctx: FastMCP context

    Returns:
        Dict with selection details

    Examples:
        # Keep only specific columns
        select_columns(session_id, ["name", "age", "email"])

        # Reorder columns by selection order
        select_columns(session_id, ["id", "date", "amount", "status"])
    """
    try:
        session, df = _get_session_data(session_id)

        # Validate columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ToolError(f"Columns not found: {missing_cols}")

        # Track counts before modification
        columns_before = len(df.columns)

        session.data_session.df = df[columns].copy()
        session.record_operation(
            OperationType.SELECT,
            {
                "columns": columns,
                "columns_before": df.columns.tolist(),
                "columns_after": columns,
            },
        )

        result = SelectColumnsResult(
            session_id=session_id,
            selected_columns=columns,
            columns_before=columns_before,
            columns_after=len(columns),
        )
        return result.model_dump()

    except Exception as e:
        logger.error(f"Error selecting columns: {e!s}")
        raise ToolError(f"Failed to select columns: {e}") from e


async def rename_columns(
    session_id: str,
    mapping: dict[str, str],
    ctx: Context | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    """Rename columns in the dataframe.

    Args:
        session_id: Session identifier
        mapping: Dict of old_name -> new_name
        ctx: FastMCP context

    Returns:
        Dict with rename details

    Examples:
        # Using dictionary mapping
        rename_columns(session_id, {"old_col1": "new_col1", "old_col2": "new_col2"})

        # Rename multiple columns
        rename_columns(session_id, {
            "FirstName": "first_name",
            "LastName": "last_name",
            "EmailAddress": "email"
        })
    """
    try:
        session, df = _get_session_data(session_id)

        # Validate columns exist
        missing_cols = [col for col in mapping if col not in df.columns]
        if missing_cols:
            raise ToolError(f"Columns not found: {missing_cols}")

        # Apply renaming
        session.data_session.df = df.rename(columns=mapping)
        session.record_operation(
            OperationType.RENAME,
            {"mapping": mapping, "renamed_count": len(mapping)},
        )

        result = RenameColumnsResult(
            session_id=session_id,
            renamed=mapping,
            columns=list(mapping.values()),
        )
        return result.model_dump()

    except Exception as e:
        logger.error(f"Error renaming columns: {e!s}")
        raise ToolError(f"Failed to rename columns: {e}") from e


async def add_column(
    session_id: str,
    name: str,
    value: CellValue | list[CellValue] = None,
    formula: str | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Add a new column to the dataframe.

    Args:
        session_id: Session identifier
        name: Name for the new column
        value: Single value for all rows, or list of values (one per row)
        formula: Python expression to compute values (e.g., "col1 + col2")
        ctx: FastMCP context

    Returns:
        ColumnOperationResult with operation details

    Examples:
        # Add column with constant value
        add_column(session_id, "status", "active")

        # Add column with list of values
        add_column(session_id, "scores", [85, 90, 78, 92, 88])

        # Add computed column
        add_column(session_id, "total", formula="price * quantity")

        # Add column with complex formula
        add_column(session_id, "full_name", formula="first_name + ' ' + last_name")
    """
    try:
        session, df = _get_session_data(session_id)

        if name in df.columns:
            raise InvalidParameterError("name", name, f"Column '{name}' already exists")

        if formula:
            try:
                # Use pandas eval to safely evaluate formula
                session.data_session.df[name] = df.eval(formula)
            except Exception as e:
                raise InvalidParameterError("formula", formula, f"Invalid formula: {e}") from e
        elif isinstance(value, list):
            if len(value) != len(df):
                raise InvalidParameterError(
                    "value",
                    str(value),
                    f"List length ({len(value)}) must match row count ({len(df)})",
                )
            session.data_session.df[name] = value
        else:
            # Single value for all rows
            session.data_session.df[name] = value

        session.record_operation(
            OperationType.ADD_COLUMN,
            {
                "column": name,
                "value_type": "formula"
                if formula
                else "list"
                if isinstance(value, list)
                else "scalar",
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="add",
            rows_affected=len(df),
            columns_affected=[name],
        )

    except (SessionNotFoundError, NoDataLoadedError, InvalidParameterError) as e:
        logger.error(f"Add column failed with {type(e).__name__}: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Unexpected error adding column: {e!s}")
        raise ToolError(f"Failed to add column: {e}") from e


async def remove_columns(
    session_id: str,
    columns: list[str],
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Remove columns from the dataframe.

    Args:
        session_id: Session identifier
        columns: List of column names to remove
        ctx: FastMCP context

    Returns:
        ColumnOperationResult with removal details

    Examples:
        # Remove single column
        remove_columns(session_id, ["temp_column"])

        # Remove multiple columns
        remove_columns(session_id, ["col1", "col2", "col3"])

        # Clean up after analysis
        remove_columns(session_id, ["_temp", "_backup", "old_value"])
    """
    try:
        session, df = _get_session_data(session_id)

        # Validate columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ColumnNotFoundError(str(missing_cols[0]), df.columns.tolist())

        session.data_session.df = df.drop(columns=columns)
        session.record_operation(
            OperationType.REMOVE_COLUMN,
            {"columns": columns, "count": len(columns)},
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="remove",
            rows_affected=len(df),
            columns_affected=columns,
        )

    except (SessionNotFoundError, NoDataLoadedError, ColumnNotFoundError) as e:
        logger.error(f"Remove columns failed with {type(e).__name__}: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Unexpected error removing columns: {e!s}")
        raise ToolError(f"Failed to remove columns: {e}") from e


async def change_column_type(
    session_id: str,
    column: str,
    dtype: Literal["int", "float", "str", "bool", "datetime"],
    errors: Literal["raise", "coerce"] = "coerce",
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Change the data type of a column.

    Args:
        session_id: Session identifier
        column: Column name to convert
        dtype: Target data type
        errors: How to handle conversion errors:
            - "raise": Raise an error if conversion fails
            - "coerce": Convert invalid values to NaN/None
        ctx: FastMCP context

    Returns:
        ColumnOperationResult with conversion details

    Examples:
        # Convert string numbers to integers
        change_column_type(session_id, "age", "int")

        # Convert to float, replacing errors with NaN
        change_column_type(session_id, "price", "float", errors="coerce")

        # Convert to datetime
        change_column_type(session_id, "date", "datetime")

        # Convert to boolean
        change_column_type(session_id, "is_active", "bool")
    """
    try:
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ColumnNotFoundError(column, df.columns.tolist())

        # Track before state
        original_dtype = str(df[column].dtype)
        null_count_before = df[column].isna().sum()

        # Map string dtype to pandas dtype
        type_map = {
            "int": "int64",
            "float": "float64",
            "str": "string",
            "bool": "bool",
            "datetime": "datetime64[ns]",
        }

        target_dtype = type_map.get(dtype)
        if not target_dtype:
            raise InvalidParameterError("dtype", dtype, f"Unsupported type: {dtype}")

        try:
            if dtype == "datetime":
                # Special handling for datetime conversion
                session.data_session.df[column] = pd.to_datetime(df[column], errors=errors)
            else:
                # General type conversion
                if errors == "coerce":
                    if dtype in ["int", "float"]:
                        session.data_session.df[column] = pd.to_numeric(df[column], errors="coerce")
                    else:
                        session.data_session.df[column] = df[column].astype(target_dtype)  # type: ignore[call-overload]
                else:
                    session.data_session.df[column] = df[column].astype(target_dtype)  # type: ignore[call-overload]

        except (ValueError, TypeError) as e:
            if errors == "raise":
                raise InvalidParameterError(
                    "column", column, f"Cannot convert to {dtype}: {e}"
                ) from e
            # If errors='coerce', the conversion has already handled invalid values

        # Track after state
        null_count_after = session.data_session.df[column].isna().sum()

        session.record_operation(
            OperationType.CHANGE_TYPE,
            {
                "column": column,
                "from_type": original_dtype,
                "to_type": dtype,
                "nulls_created": int(null_count_after - null_count_before),
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation=f"change_type_to_{dtype}",
            rows_affected=len(df),
            columns_affected=[column],
        )

    except (
        SessionNotFoundError,
        NoDataLoadedError,
        ColumnNotFoundError,
        InvalidParameterError,
    ) as e:
        logger.error(f"Change column type failed with {type(e).__name__}: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Unexpected error changing column type: {e!s}")
        raise ToolError(f"Failed to change column type: {e}") from e


async def update_column(
    session_id: str,
    column: str,
    operation: Literal["replace", "map", "apply", "fillna"],
    value: Any | None = None,
    pattern: str | None = None,
    replacement: str | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Update values in a column using various operations.

    Args:
        session_id: Session identifier
        column: Column name to update
        operation: Type of update operation:
            - "replace": Replace pattern with replacement
            - "map": Map values using a dictionary
            - "apply": Apply a function/expression
            - "fillna": Fill missing values
        value: Value for the operation (depends on operation type)
        pattern: Pattern for replace operation
        replacement: Replacement for replace operation
        ctx: FastMCP context

    Returns:
        ColumnOperationResult with update details

    Examples:
        # Replace values
        update_column(session_id, "status", "replace", pattern="N/A", replacement="Unknown")

        # Map values
        update_column(session_id, "code", "map", value={"A": "Alpha", "B": "Beta"})

        # Fill missing values
        update_column(session_id, "score", "fillna", value=0)
    """
    try:
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ColumnNotFoundError(column, df.columns.tolist())

        # Track initial state
        null_count_before = df[column].isna().sum()

        if operation == "replace":
            if pattern is None or replacement is None:
                raise InvalidParameterError(
                    "pattern/replacement",
                    f"{pattern}/{replacement}",
                    "Both pattern and replacement required for replace operation",
                )
            session.data_session.df[column] = df[column].replace(pattern, replacement)

        elif operation == "map":
            if not isinstance(value, dict):
                raise InvalidParameterError(
                    "value",
                    str(value),
                    "Dictionary mapping required for map operation",
                )
            session.data_session.df[column] = df[column].map(value)

        elif operation == "apply":
            if value is None:
                raise InvalidParameterError(
                    "value",
                    str(value),
                    "Expression or function required for apply operation",
                )
            # For simple expressions, use eval safely with restricted scope
            if isinstance(value, str):
                import ast

                # Parse and validate the expression is safe
                try:
                    ast.parse(value, mode="eval")
                    # Create a safe evaluation context
                    safe_dict: dict[str, Any] = {"x": None, "__builtins__": {}}
                    session.data_session.df[column] = df[column].apply(
                        lambda x: eval(value, safe_dict, {"x": x})  # noqa: S307
                    )
                except SyntaxError as e:
                    raise InvalidParameterError("value", value, f"Invalid expression: {e}") from e
            else:
                session.data_session.df[column] = df[column].apply(value)

        elif operation == "fillna":
            if value is None:
                raise InvalidParameterError(
                    "value",
                    str(value),
                    "Fill value required for fillna operation",
                )
            session.data_session.df[column] = df[column].fillna(value)

        else:
            raise InvalidParameterError(
                "operation",
                operation,
                "Supported operations: replace, map, apply, fillna",
            )

        # Track changes
        null_count_after = session.data_session.df[column].isna().sum()

        session.record_operation(
            OperationType.UPDATE_COLUMN,
            {
                "column": column,
                "operation": operation,
                "nulls_changed": int(null_count_after - null_count_before),
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation=f"update_{operation}",
            rows_affected=len(df),
            columns_affected=[column],
        )

    except (
        SessionNotFoundError,
        NoDataLoadedError,
        ColumnNotFoundError,
        InvalidParameterError,
    ) as e:
        logger.error(f"Update column failed with {type(e).__name__}: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Unexpected error updating column: {e!s}")
        raise ToolError(f"Failed to update column: {e}") from e


# =============================================================================
# SERVER INITIALIZATION
# =============================================================================

column_server = FastMCP(
    "DataBeak Column Operations Server",
    instructions="Column-level operations server providing selection, renaming, addition, removal, and type conversion",
)

# Register the functions as MCP tools
column_server.tool(name="select_columns")(select_columns)
column_server.tool(name="rename_columns")(rename_columns)
column_server.tool(name="add_column")(add_column)
column_server.tool(name="remove_columns")(remove_columns)
column_server.tool(name="change_column_type")(change_column_type)
column_server.tool(name="update_column")(update_column)
