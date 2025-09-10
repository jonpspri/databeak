"""FastMCP server for column-level data operations.

This server provides column selection, renaming, addition, removal, and type conversion.
"""

from __future__ import annotations

from typing import Any, Literal

from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from ..models.tool_responses import ColumnOperationResult
from ..tools.transformations import add_column as _add_column
from ..tools.transformations import change_column_type as _change_column_type
from ..tools.transformations import remove_columns as _remove_columns
from ..tools.transformations import rename_columns as _rename_columns
from ..tools.transformations import select_columns as _select_columns
from ..tools.transformations import update_column as _update_column

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
# SERVER INITIALIZATION
# =============================================================================

column_server = FastMCP("DataBeak Column Operations Server")


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================


@column_server.tool
async def select_columns(
    session_id: str, columns: list[str], ctx: Context | None = None
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
    result = await _select_columns(session_id, columns, ctx)
    return result.model_dump()


@column_server.tool
async def rename_columns(
    session_id: str,
    mapping: dict[str, str] | list[ColumnMapping],
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Rename columns in the dataframe.

    Args:
        session_id: Session identifier
        mapping: Dict of old_name -> new_name or list of ColumnMapping objects
        ctx: FastMCP context

    Returns:
        Dict with rename details

    Examples:
        # Using dictionary mapping
        rename_columns(session_id, {"old_col1": "new_col1", "old_col2": "new_col2"})

        # Using ColumnMapping models
        rename_columns(session_id, [
            ColumnMapping(old_name="FirstName", new_name="first_name"),
            ColumnMapping(old_name="LastName", new_name="last_name")
        ])
    """
    # Convert ColumnMapping list to dict if needed
    if isinstance(mapping, list):
        mapping_dict = {m.old_name: m.new_name for m in mapping}
    else:
        mapping_dict = mapping

    result = await _rename_columns(session_id, mapping_dict, ctx)
    return result.model_dump()


@column_server.tool
async def add_column(
    session_id: str,
    name: str,
    value: CellValue | list[CellValue] = None,
    formula: str | None = None,
    ctx: Context | None = None,
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
    return await _add_column(session_id, name, value, formula, ctx)


@column_server.tool
async def remove_columns(
    session_id: str, columns: list[str], ctx: Context | None = None
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
    return await _remove_columns(session_id, columns, ctx)


@column_server.tool
async def change_column_type(
    session_id: str,
    column: str,
    dtype: Literal["int", "float", "str", "bool", "datetime"],
    errors: Literal["raise", "coerce"] = "coerce",
    ctx: Context | None = None,
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
    return await _change_column_type(session_id, column, dtype, errors, ctx)


@column_server.tool
async def update_column(
    session_id: str,
    column: str,
    operation: Literal["replace", "map", "apply", "fillna"],
    value: Any | None = None,
    pattern: str | None = None,
    replacement: str | None = None,
    ctx: Context | None = None,
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
    return await _update_column(session_id, column, operation, value, pattern, replacement, ctx)
