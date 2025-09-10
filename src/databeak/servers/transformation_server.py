"""FastMCP server for core data transformation operations.

This server provides filtering, sorting, deduplication, and missing value handling.
"""

from __future__ import annotations

from typing import Any, Literal

from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from ..models.tool_responses import ColumnOperationResult, FilterOperationResult
from ..tools.transformations import fill_missing_values as _fill_missing_values
from ..tools.transformations import filter_rows as _filter_rows
from ..tools.transformations import remove_duplicates as _remove_duplicates
from ..tools.transformations import sort_data as _sort_data

# Type aliases
CellValue = str | int | float | bool | None

# =============================================================================
# PYDANTIC MODELS FOR REQUEST PARAMETERS
# =============================================================================


class FilterCondition(BaseModel):
    """Filter condition for row filtering."""

    column: str = Field(description="Column name to filter on")
    operator: Literal[
        "==",
        "!=",
        ">",
        "<",
        ">=",
        "<=",
        "contains",
        "not_contains",
        "starts_with",
        "ends_with",
        "in",
        "not_in",
        "is_null",
        "is_not_null",
    ] = Field(description="Comparison operator")
    value: CellValue | list[CellValue] = Field(
        default=None, description="Value to compare against (not needed for null operators)"
    )


class SortColumn(BaseModel):
    """Column specification for sorting."""

    column: str = Field(description="Column name to sort by")
    ascending: bool = Field(default=True, description="Sort in ascending order")


# =============================================================================
# TOOL DEFINITIONS (Direct implementations for testing)
# =============================================================================


async def filter_rows(
    session_id: str,
    conditions: list[FilterCondition | dict[str, Any]],
    mode: Literal["and", "or"] = "and",
    ctx: Context | None = None,
) -> FilterOperationResult:
    """Filter rows using flexible conditions with comprehensive null value and text matching
    support.

    Provides powerful filtering capabilities optimized for AI-driven data analysis. Supports
    multiple operators, logical combinations, and comprehensive null value handling.

    Args:
        session_id: Session identifier for the active CSV data session
        conditions: List of filter conditions with column, operator, and value
        mode: Logic for combining conditions ("and" or "or")

    Returns:
        FilterOperationResult with filtering statistics

    Examples:
        # Numeric filtering
        filter_rows(session_id, [{"column": "age", "operator": ">", "value": 25}])

        # Text filtering with null handling
        filter_rows(session_id, [
            {"column": "name", "operator": "contains", "value": "Smith"},
            {"column": "email", "operator": "is_not_null"}
        ], mode="and")

        # Multiple conditions with OR logic
        filter_rows(session_id, [
            {"column": "status", "operator": "==", "value": "active"},
            {"column": "priority", "operator": "==", "value": "high"}
        ], mode="or")
    """
    # Convert Pydantic models to dicts for compatibility with existing implementation
    condition_dicts = []
    for c in conditions:
        if isinstance(c, FilterCondition):
            condition_dicts.append(c.model_dump())
        else:
            condition_dicts.append(c)
    return await _filter_rows(session_id, condition_dicts, mode, ctx)


async def sort_data(
    session_id: str,
    columns: list[str | SortColumn | dict[str, Any]],
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Sort data by one or more columns.

    Args:
        session_id: Session identifier
        columns: Column names (strings) or SortColumn specifications
        ctx: FastMCP context

    Returns:
        Dict with sorting details

    Examples:
        # Simple single column sort
        sort_data(session_id, ["age"])

        # Multi-column sort
        sort_data(session_id, ["department", "salary"])

        # Using dicts for precise control
        sort_data(session_id, [
            {"column": "department", "ascending": True},
            {"column": "salary", "ascending": False}
        ])
    """
    # Convert to the format expected by the underlying function
    column_list: list[str | dict[str, str]] = []
    for col in columns:
        if isinstance(col, SortColumn):
            column_list.append({"column": col.column, "ascending": str(col.ascending)})
        elif isinstance(col, dict):
            # Pass dict through, converting ascending to string if present
            if "ascending" in col:
                column_list.append({"column": col["column"], "ascending": str(col["ascending"])})
            else:
                column_list.append(col)
        else:
            column_list.append(col)

    result = await _sort_data(session_id, column_list, ctx)
    return result.model_dump()


async def remove_duplicates(
    session_id: str,
    subset: list[str] | None = None,
    keep: Literal["first", "last", "none"] = "first",
    ctx: Context | None = None,
) -> ColumnOperationResult:
    """Remove duplicate rows from the dataframe.

    Args:
        session_id: Session identifier
        subset: Columns to consider for duplicates (None = all columns)
        keep: Which duplicates to keep ("first", "last", or "none" to drop all)
        ctx: FastMCP context

    Returns:
        ColumnOperationResult with duplicate removal statistics

    Examples:
        # Remove exact duplicate rows
        remove_duplicates(session_id)

        # Remove duplicates based on specific columns
        remove_duplicates(session_id, subset=["email", "name"])

        # Keep last occurrence instead of first
        remove_duplicates(session_id, subset=["id"], keep="last")

        # Remove all duplicates (keep none)
        remove_duplicates(session_id, subset=["email"], keep="none")
    """
    return await _remove_duplicates(session_id, subset, keep, ctx)


async def fill_missing_values(
    session_id: str,
    strategy: Literal["drop", "fill", "forward", "backward", "mean", "median", "mode"] = "drop",
    value: Any = None,
    columns: list[str] | None = None,
    ctx: Context | None = None,
) -> ColumnOperationResult:
    """Fill or remove missing values in the dataframe.

    Args:
        session_id: Session identifier
        strategy: Strategy for handling missing values:
            - "drop": Remove rows with missing values
            - "fill": Fill with a specific value
            - "forward": Forward fill (use previous valid value)
            - "backward": Backward fill (use next valid value)
            - "mean": Fill with column mean (numeric only)
            - "median": Fill with column median (numeric only)
            - "mode": Fill with most common value
        value: Value to use when strategy is "fill"
        columns: Columns to process (None = all columns)
        ctx: FastMCP context

    Returns:
        ColumnOperationResult with operation statistics

    Examples:
        # Drop rows with any missing values
        fill_missing_values(session_id, strategy="drop")

        # Fill missing values with 0
        fill_missing_values(session_id, strategy="fill", value=0)

        # Forward fill specific columns
        fill_missing_values(session_id, strategy="forward", columns=["price", "quantity"])

        # Fill with column mean for numeric columns
        fill_missing_values(session_id, strategy="mean", columns=["age", "salary"])
    """
    return await _fill_missing_values(session_id, strategy, value, columns, ctx)


# =============================================================================
# SERVER INITIALIZATION
# =============================================================================

transformation_server = FastMCP(
    "DataBeak Transformation Server",
    instructions="Core data transformation server providing filtering, sorting, deduplication, and missing value handling",
)

# Register the functions as MCP tools
transformation_server.tool(name="filter_rows")(filter_rows)
transformation_server.tool(name="sort_data")(sort_data)
transformation_server.tool(name="remove_duplicates")(remove_duplicates)
transformation_server.tool(name="fill_missing_values")(fill_missing_values)
