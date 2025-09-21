"""Standalone transformation server for DataBeak using FastMCP server composition."""

from __future__ import annotations

import logging
from typing import Annotated, Literal

import pandas as pd
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

# Import session management from the main package
from ..models import get_session_manager
from ..models.tool_responses import ColumnOperationResult, FilterOperationResult, SortDataResult
from ..models.typed_dicts import FilterConditionDict

logger = logging.getLogger(__name__)

# Type aliases
CellValue = str | int | float | bool | None

# ============================================================================
# PYDANTIC MODELS FOR REQUEST PARAMETERS
# ============================================================================


# Use existing FilterCondition from data_models instead of defining locally


class SortColumn(BaseModel):
    """Column specification for sorting."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column name to sort by")
    ascending: bool = Field(default=True, description="Sort in ascending order")


# ============================================================================
# TRANSFORMATION LOGIC (Direct implementations)
# ============================================================================


def filter_rows(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    conditions: Annotated[
        list[FilterConditionDict],
        Field(description="List of filter conditions with column, operator, and value"),
    ],
    mode: Annotated[
        Literal["and", "or"],
        Field(description="Logic for combining conditions (and/or)"),
    ] = "and",
) -> FilterOperationResult:
    """Filter rows using flexible conditions: comprehensive null value and text matching support.

    Provides powerful filtering capabilities optimized for AI-driven data analysis. Supports
    multiple operators, logical combinations, and comprehensive null value handling.

    Examples:
        # Numeric filtering
        filter_rows(ctx, [{"column": "age", "operator": ">", "value": 25}])

        # Text filtering with null handling
        filter_rows(ctx, [
            {"column": "name", "operator": "contains", "value": "Smith"},
            {"column": "email", "operator": "is_not_null"}
        ], mode="and")

        # Multiple conditions with OR logic
        filter_rows(ctx, [
            {"column": "status", "operator": "==", "value": "active"},
            {"column": "priority", "operator": "==", "value": "high"}
        ], mode="or")

    """
    try:
        session_id = ctx.session_id
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)

        if not session or session.df is None:
            msg = "Invalid session or no data loaded"
            raise ToolError(msg)

        df = session.df
        rows_before = len(df)

        # Initialize mask based on mode: AND starts True, OR starts False
        mask = pd.Series([mode == "and"] * len(df))

        # Process conditions (handles both FilterConditionDict and FilterCondition objects)
        for condition in conditions:
            if isinstance(condition, dict):
                # It's a FilterConditionDict
                column = condition.get("column")
                operator = condition.get("operator")
                value = condition.get("value")
            else:
                # It's a FilterCondition object
                column = condition.column
                operator = condition.operator
                value = condition.value

            if column is None or column not in df.columns:
                msg = f"Column '{column}' not found in data"
                raise ToolError(msg)

            col_data = df[column]

            if operator == "=" or operator == "==":
                condition_mask = col_data == value
            elif operator == "!=" or operator == "not_equals":
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
                msg = (
                    f"Invalid operator '{operator}'. Valid operators: "
                    "==, !=, >, <, >=, <=, contains, not_contains, starts_with, ends_with, "
                    "in, not_in, is_null, is_not_null"
                )
                raise ToolError(
                    msg,
                )

            mask = mask & condition_mask if mode == "and" else mask | condition_mask

        # Apply filter
        session.df = df[mask].reset_index(drop=True)
        rows_after = len(session.df)

        # Convert conditions to JSON-serializable format for history
        serializable_conditions = []
        for condition in conditions:
            if isinstance(condition, dict):
                # It's already a dict
                serializable_conditions.append(dict(condition))
            else:
                # It's a FilterCondition object - convert to dict
                serializable_conditions.append(
                    {
                        "column": condition.column,
                        "operator": condition.operator,
                        "value": condition.value,
                    },
                )

        # No longer recording operations (simplified MCP architecture)

        return FilterOperationResult(
            rows_before=rows_before,
            rows_after=rows_after,
            rows_filtered=rows_before - rows_after,
            conditions_applied=len(conditions),
        )

    except Exception as e:
        logger.error("Error filtering rows: %s", str(e))
        msg = f"Error filtering rows: {e!s}"
        raise ToolError(msg) from e


def sort_data(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    columns: Annotated[
        list[str | SortColumn],
        Field(description="Column specifications for sorting (strings or SortColumn objects)"),
    ],
) -> SortDataResult:
    """Sort data by one or more columns with comprehensive error handling.

    Provides flexible sorting capabilities with support for multiple columns
    and sort directions. Handles mixed data types appropriately and maintains
    data integrity throughout the sorting process.

    Examples:
        # Simple single column sort
        sort_data(ctx, ["age"])

        # Multi-column sort with different directions
        sort_data(ctx, [
            {"column": "department", "ascending": True},
            {"column": "salary", "ascending": False}
        ])

        # Using SortColumn objects for type safety
        sort_data(ctx, [
            SortColumn(column="name", ascending=True),
            SortColumn(column="age", ascending=False)
        ])
    """
    try:
        session_id = ctx.session_id
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)

        if not session or session.df is None:
            msg = "Invalid session or no data loaded"
            raise ToolError(msg)

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
            elif isinstance(col, dict) and "column" in col:
                sort_columns.append(col["column"])
                ascending.append(col.get("ascending", True))
            else:
                msg = f"Invalid column specification: {col}"
                raise ToolError(msg)

        # Validate all columns exist
        missing_cols = [col for col in sort_columns if col not in df.columns]
        if missing_cols:
            msg = f"Columns not found: {missing_cols}"
            raise ToolError(msg)

        # Perform sort
        session.df = df.sort_values(by=sort_columns, ascending=ascending).reset_index(drop=True)

        # No longer recording operations (simplified MCP architecture)

        return SortDataResult(
            sorted_by=sort_columns,
            ascending=ascending,
            rows_processed=len(df),
        )

    except Exception as e:
        logger.error("Error sorting data: %s", str(e))
        msg = f"Error sorting data: {e!s}"
        raise ToolError(msg) from e


def remove_duplicates(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    subset: Annotated[
        list[str] | None,
        Field(description="Columns to consider for duplicates (None = all columns)"),
    ] = None,
    keep: Annotated[
        Literal["first", "last", "none"],
        Field(description="Which duplicates to keep: first, last, or none"),
    ] = "first",
) -> ColumnOperationResult:
    """Remove duplicate rows from the dataframe with comprehensive validation.

    Provides flexible duplicate removal with options for column subset selection
    and different keep strategies. Handles edge cases and provides detailed
    statistics about the deduplication process.

    Examples:
        # Remove exact duplicate rows
        remove_duplicates(ctx)

        # Remove duplicates based on specific columns
        remove_duplicates(ctx, subset=["email", "name"])

        # Keep last occurrence instead of first
        remove_duplicates(ctx, subset=["id"], keep="last")

        # Remove all duplicates (keep none)
        remove_duplicates(ctx, subset=["email"], keep="none")
    """
    try:
        session_id = ctx.session_id
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)

        if not session or session.df is None:
            msg = "Invalid session or no data loaded"
            raise ToolError(msg)

        df = session.df
        rows_before = len(df)

        # Validate subset columns if provided
        if subset:
            missing_cols = [col for col in subset if col not in df.columns]
            if missing_cols:
                msg = f"Columns not found in subset: {missing_cols}"
                raise ToolError(msg)

        # Convert keep parameter for pandas
        keep_param: Literal["first", "last"] | Literal[False] = keep if keep != "none" else False

        # Remove duplicates
        session.df = df.drop_duplicates(subset=subset, keep=keep_param).reset_index(drop=True)

        rows_after = len(session.df)
        rows_removed = rows_before - rows_after

        # No longer recording operations (simplified MCP architecture)

        return ColumnOperationResult(
            operation="remove_duplicates",
            rows_affected=rows_after,
            columns_affected=subset if subset else df.columns.tolist(),
            rows_removed=rows_removed,
        )

    except Exception as e:
        logger.error("Error removing duplicates: %s", str(e))
        msg = f"Error removing duplicates: {e!s}"
        raise ToolError(msg) from e


def fill_missing_values(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    strategy: Annotated[
        Literal["drop", "fill", "forward", "backward", "mean", "median", "mode"],
        Field(
            description="Strategy for handling missing values (drop, fill, forward, backward, mean, median, mode)",
        ),
    ] = "drop",
    value: Annotated[CellValue, Field(description="Value to use when strategy is 'fill'")] = None,
    columns: Annotated[
        list[str] | None,
        Field(description="Columns to process (None = all columns)"),
    ] = None,
) -> ColumnOperationResult:
    """Fill or remove missing values with comprehensive strategy support.

    Provides multiple strategies for handling missing data, including statistical
    imputation methods. Handles different data types appropriately and validates
    strategy compatibility with column types.

    Examples:
        # Drop rows with any missing values
        fill_missing_values(ctx, strategy="drop")

        # Fill missing values with 0
        fill_missing_values(ctx, strategy="fill", value=0)

        # Forward fill specific columns
        fill_missing_values(ctx, strategy="forward", columns=["price", "quantity"])

        # Fill with column mean for numeric columns
        fill_missing_values(ctx, strategy="mean", columns=["age", "salary"])
    """
    try:
        session_id = ctx.session_id
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)

        if not session or session.df is None:
            msg = "Invalid session or no data loaded"
            raise ToolError(msg)

        df = session.df

        # Validate and set target columns
        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                msg = f"Columns not found: {missing_cols}"
                raise ToolError(msg)
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
                msg = "Value required for 'fill' strategy"
                raise ToolError(msg)
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
                    logger.warning("Column '%s' is not numeric, skipping mean fill", col)
        elif strategy == "median":
            session.df = df.copy()
            for col in target_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    if not pd.isna(median_val):
                        session.df[col] = df[col].fillna(median_val)
                else:
                    logger.warning("Column '%s' is not numeric, skipping median fill", col)
        elif strategy == "mode":
            session.df = df.copy()
            for col in target_cols:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    session.df[col] = df[col].fillna(mode_val[0])
        else:
            msg = (
                f"Invalid strategy '{strategy}'. Valid strategies: "
                "drop, fill, forward, backward, mean, median, mode"
            )
            raise ToolError(
                msg,
            )

        rows_after = len(session.df)
        missing_after = session.df[target_cols].isna().sum().sum()
        values_filled = missing_before - missing_after

        # No longer recording operations (simplified MCP architecture)

        return ColumnOperationResult(
            operation="fill_missing_values",
            rows_affected=rows_after,
            columns_affected=target_cols,
            values_filled=int(values_filled),
        )

    except Exception as e:
        logger.error("Error filling missing values: %s", str(e))
        msg = f"Error filling missing values: {e!s}"
        raise ToolError(msg) from e


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
