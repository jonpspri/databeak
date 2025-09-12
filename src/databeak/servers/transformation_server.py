"""Standalone transformation server for DataBeak using FastMCP server composition."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

import pandas as pd
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

# Import session management from the main package
from ..models import OperationType, get_session_manager
from ..models.tool_responses import ColumnOperationResult, FilterOperationResult, SortDataResult

logger = logging.getLogger(__name__)

# Type aliases
CellValue = str | int | float | bool | None

# ============================================================================
# PYDANTIC MODELS FOR REQUEST PARAMETERS
# ============================================================================


class FilterCondition(BaseModel):
    """Filter condition for row filtering."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column name to sort by")
    ascending: bool = Field(default=True, description="Sort in ascending order")


# ============================================================================
# TRANSFORMATION LOGIC (Direct implementations)
# ============================================================================


def filter_rows(
    session_id: Annotated[str, Field(description="Session identifier containing the target data")],
    conditions: Annotated[
        list[FilterCondition | dict[str, Any]],
        Field(description="List of filter conditions with column, operator, and value"),
    ],
    mode: Annotated[
        Literal["and", "or"], Field(description="Logic for combining conditions (and/or)")
    ] = "and",
    ctx: Annotated[
        Context | None, Field(description="FastMCP context for progress reporting")
    ] = None,  # noqa: ARG001
) -> FilterOperationResult:
    """Filter rows using flexible conditions with comprehensive null value and text matching
    support.

    Provides powerful filtering capabilities optimized for AI-driven data analysis. Supports
    multiple operators, logical combinations, and comprehensive null value handling.

    Args:
        session_id: Session identifier for the active CSV data session
        conditions: List of filter conditions with column, operator, and value
        mode: Logic for combining conditions ("and" or "or")
        ctx: FastMCP context

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
    try:
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session or session.df is None:
            raise ToolError("Invalid session or no data loaded")

        df = session.df
        rows_before = len(df)

        # Initialize mask based on mode: AND starts True, OR starts False
        mask = pd.Series([mode == "and"] * len(df))

        # Process conditions - convert Pydantic models if needed
        processed_conditions = []
        for condition in conditions:
            if isinstance(condition, FilterCondition):
                processed_conditions.append(condition.model_dump())
            else:
                processed_conditions.append(condition)

        for condition in processed_conditions:
            column = condition.get("column")
            operator = condition.get("operator")
            value = condition.get("value")

            if column is None or column not in df.columns:
                raise ToolError(f"Column '{column}' not found in data")

            col_data = df[column]

            if operator == "==":
                condition_mask = col_data == value
            elif operator == "!=":
                condition_mask = col_data != value
            elif operator == ">":
                condition_mask = col_data > value
            elif operator == "<":
                condition_mask = col_data < value
            elif operator == ">=":
                condition_mask = col_data >= value
            elif operator == "<=":
                condition_mask = col_data <= value
            elif operator == "contains":
                condition_mask = col_data.astype(str).str.contains(str(value), na=False)
            elif operator == "not_contains":
                condition_mask = ~col_data.astype(str).str.contains(str(value), na=False)
            elif operator == "starts_with":
                condition_mask = col_data.astype(str).str.startswith(str(value), na=False)
            elif operator == "ends_with":
                condition_mask = col_data.astype(str).str.endswith(str(value), na=False)
            elif operator == "in":
                condition_mask = col_data.isin(value if isinstance(value, list) else [value])
            elif operator == "not_in":
                condition_mask = ~col_data.isin(value if isinstance(value, list) else [value])
            elif operator == "is_null":
                condition_mask = col_data.isna()
            elif operator == "is_not_null":
                condition_mask = col_data.notna()
            else:
                raise ToolError(
                    f"Invalid operator '{operator}'. Valid operators: "
                    "==, !=, >, <, >=, <=, contains, not_contains, starts_with, ends_with, "
                    "in, not_in, is_null, is_not_null"
                )

            mask = mask & condition_mask if mode == "and" else mask | condition_mask

        # Apply filter
        session.df = df[mask].reset_index(drop=True)
        rows_after = len(session.df)

        # Record operation
        session.record_operation(
            OperationType.FILTER,
            {
                "conditions": processed_conditions,
                "mode": mode,
                "rows_before": rows_before,
                "rows_after": rows_after,
            },
        )

        return FilterOperationResult(
            session_id=session_id,
            rows_before=rows_before,
            rows_after=rows_after,
            rows_filtered=rows_before - rows_after,
            conditions_applied=len(processed_conditions),
        )

    except Exception as e:
        logger.error(f"Error filtering rows: {e!s}")
        raise ToolError(f"Error filtering rows: {e!s}") from e


def sort_data(
    session_id: Annotated[str, Field(description="Session identifier containing the target data")],
    columns: Annotated[
        list[str | SortColumn | dict[str, Any]],
        Field(
            description="Column specifications for sorting (strings, SortColumn objects, or dicts)"
        ),
    ],
    ctx: Annotated[
        Context | None, Field(description="FastMCP context for progress reporting")
    ] = None,  # noqa: ARG001
) -> SortDataResult:
    """Sort data by one or more columns with comprehensive error handling.

    Provides flexible sorting capabilities with support for multiple columns
    and sort directions. Handles mixed data types appropriately and maintains
    data integrity throughout the sorting process.

    Args:
        session_id: Session identifier for the active CSV data session
        columns: Column specifications - can be strings, SortColumn objects, or dicts
        ctx: FastMCP context

    Returns:
        SortDataResult with sorting details and statistics

    Examples:
        # Simple single column sort
        sort_data(session_id, ["age"])

        # Multi-column sort with different directions
        sort_data(session_id, [
            {"column": "department", "ascending": True},
            {"column": "salary", "ascending": False}
        ])

        # Using SortColumn objects for type safety
        sort_data(session_id, [
            SortColumn(column="name", ascending=True),
            SortColumn(column="age", ascending=False)
        ])
    """
    try:
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session or session.df is None:
            raise ToolError("Invalid session or no data loaded")

        df = session.df

        # Parse columns into names and ascending flags
        sort_columns: list[str] = []
        ascending: list[bool] = []

        for col in columns:
            if isinstance(col, str):
                sort_columns.append(col)
                ascending.append(True)
            elif isinstance(col, SortColumn):
                sort_columns.append(col.column)
                ascending.append(col.ascending)
            elif isinstance(col, dict):
                if "column" not in col:
                    raise ToolError(f"Dict specification missing 'column' key: {col}")
                sort_columns.append(col["column"])
                # Handle string/bool conversion for ascending
                asc_val = col.get("ascending", True)
                if isinstance(asc_val, str):
                    ascending.append(asc_val.lower() in ("true", "1", "yes"))
                else:
                    ascending.append(bool(asc_val))
            else:
                raise ToolError(f"Invalid column specification: {col}")

        # Validate all columns exist
        missing_cols = [col for col in sort_columns if col not in df.columns]
        if missing_cols:
            raise ToolError(f"Columns not found: {missing_cols}")

        # Perform sort
        session.df = df.sort_values(by=sort_columns, ascending=ascending).reset_index(drop=True)

        # Record operation
        session.record_operation(
            OperationType.SORT,
            {
                "columns": sort_columns,
                "ascending": ascending,
                "rows_processed": len(df),
            },
        )

        return SortDataResult(
            session_id=session_id,
            sorted_by=sort_columns,
            ascending=ascending,
            rows_processed=len(df),
        )

    except Exception as e:
        logger.error(f"Error sorting data: {e!s}")
        raise ToolError(f"Error sorting data: {e!s}") from e


def remove_duplicates(
    session_id: Annotated[str, Field(description="Session identifier containing the target data")],
    subset: Annotated[
        list[str] | None,
        Field(description="Columns to consider for duplicates (None = all columns)"),
    ] = None,
    keep: Annotated[
        Literal["first", "last", "none"],
        Field(description="Which duplicates to keep: first, last, or none"),
    ] = "first",
    ctx: Annotated[
        Context | None, Field(description="FastMCP context for progress reporting")
    ] = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Remove duplicate rows from the dataframe with comprehensive validation.

    Provides flexible duplicate removal with options for column subset selection
    and different keep strategies. Handles edge cases and provides detailed
    statistics about the deduplication process.

    Args:
        session_id: Session identifier for the active CSV data session
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
    try:
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session or session.df is None:
            raise ToolError("Invalid session or no data loaded")

        df = session.df
        rows_before = len(df)

        # Validate subset columns if provided
        if subset:
            missing_cols = [col for col in subset if col not in df.columns]
            if missing_cols:
                raise ToolError(f"Columns not found in subset: {missing_cols}")

        # Convert keep parameter for pandas
        keep_param: Literal["first", "last"] | Literal[False] = keep if keep != "none" else False

        # Remove duplicates
        session.df = df.drop_duplicates(subset=subset, keep=keep_param).reset_index(drop=True)

        rows_after = len(session.df)
        rows_removed = rows_before - rows_after

        # Record operation
        session.record_operation(
            OperationType.REMOVE_DUPLICATES,
            {
                "subset": subset,
                "keep": keep,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_removed": rows_removed,
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="remove_duplicates",
            rows_affected=rows_after,
            columns_affected=subset if subset else df.columns.tolist(),
            rows_removed=rows_removed,
        )

    except Exception as e:
        logger.error(f"Error removing duplicates: {e!s}")
        raise ToolError(f"Error removing duplicates: {e!s}") from e


def fill_missing_values(
    session_id: Annotated[str, Field(description="Session identifier containing the target data")],
    strategy: Annotated[
        Literal["drop", "fill", "forward", "backward", "mean", "median", "mode"],
        Field(
            description="Strategy for handling missing values (drop, fill, forward, backward, mean, median, mode)"
        ),
    ] = "drop",
    value: Annotated[Any, Field(description="Value to use when strategy is 'fill'")] = None,
    columns: Annotated[
        list[str] | None, Field(description="Columns to process (None = all columns)")
    ] = None,
    ctx: Annotated[
        Context | None, Field(description="FastMCP context for progress reporting")
    ] = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Fill or remove missing values with comprehensive strategy support.

    Provides multiple strategies for handling missing data, including statistical
    imputation methods. Handles different data types appropriately and validates
    strategy compatibility with column types.

    Args:
        session_id: Session identifier for the active CSV data session
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
    try:
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session or session.df is None:
            raise ToolError("Invalid session or no data loaded")

        df = session.df
        rows_before = len(df)

        # Validate and set target columns
        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ToolError(f"Columns not found: {missing_cols}")
            target_cols = columns
        else:
            target_cols = df.columns.tolist()

        # Count missing values before processing
        missing_before = df[target_cols].isna().sum().sum()

        # Apply strategy
        if strategy == "drop":
            session.df = df.dropna(subset=target_cols)
        elif strategy == "fill":
            if value is None:
                raise ToolError("Value required for 'fill' strategy")
            session.df = df.copy()
            session.df[target_cols] = df[target_cols].fillna(value)
        elif strategy == "forward":
            session.df = df.copy()
            session.df[target_cols] = df[target_cols].ffill()
        elif strategy == "backward":
            session.df = df.copy()
            session.df[target_cols] = df[target_cols].bfill()
        elif strategy == "mean":
            session.df = df.copy()
            for col in target_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    mean_val = df[col].mean()
                    if not pd.isna(mean_val):
                        session.df[col] = df[col].fillna(mean_val)
                else:
                    logger.warning(f"Column '{col}' is not numeric, skipping mean fill")
        elif strategy == "median":
            session.df = df.copy()
            for col in target_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    if not pd.isna(median_val):
                        session.df[col] = df[col].fillna(median_val)
                else:
                    logger.warning(f"Column '{col}' is not numeric, skipping median fill")
        elif strategy == "mode":
            session.df = df.copy()
            for col in target_cols:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    session.df[col] = df[col].fillna(mode_val[0])
        else:
            raise ToolError(
                f"Invalid strategy '{strategy}'. Valid strategies: "
                "drop, fill, forward, backward, mean, median, mode"
            )

        rows_after = len(session.df)
        missing_after = session.df[target_cols].isna().sum().sum()
        values_filled = missing_before - missing_after

        # Record operation
        session.record_operation(
            OperationType.FILL_MISSING,
            {
                "strategy": strategy,
                "value": str(value) if value is not None else None,
                "columns": target_cols,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "values_filled": int(values_filled),
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="fill_missing_values",
            rows_affected=rows_after,
            columns_affected=target_cols,
            values_filled=int(values_filled),
        )

    except Exception as e:
        logger.error(f"Error filling missing values: {e!s}")
        raise ToolError(f"Error filling missing values: {e!s}") from e


# ============================================================================
# FASTMCP SERVER SETUP
# ============================================================================


# Create transformation server
transformation_server = FastMCP(
    "DataBeak-Transformation",
    instructions="Core data transformation server for filtering, sorting, deduplication, and missing value handling",
)

# Register the logic functions directly as MCP tools (no wrapper functions needed)
transformation_server.tool(name="filter_rows")(filter_rows)
transformation_server.tool(name="sort_data")(sort_data)
transformation_server.tool(name="remove_duplicates")(remove_duplicates)
transformation_server.tool(name="fill_missing_values")(fill_missing_values)
