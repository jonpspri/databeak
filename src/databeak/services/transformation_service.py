"""Data transformation operations with filtering, sorting, and column manipulation."""

from __future__ import annotations

import logging
from typing import Any, Literal

import pandas as pd
from fastmcp.exceptions import ToolError

from ..exceptions import (
    ColumnNotFoundError,
    InvalidParameterError,
    NoDataLoadedError,
    SessionNotFoundError,
)
from ..models import OperationType, get_session_manager
from ..models.tool_responses import BaseToolResponse

logger = logging.getLogger(__name__)

# ============================================================================
# TYPE ALIASES
# ============================================================================

CsvCellValue = str | int | float | bool | None
RowData = dict[str, CsvCellValue] | list[CsvCellValue]

# ============================================================================
# PYDANTIC MODELS FOR TRANSFORMATION OPERATIONS
# ============================================================================


class TransformationResult(BaseToolResponse):
    """Base transformation operation response."""

    session_id: str
    operation: str
    rows_affected: int
    columns_affected: list[str] | None = None
    message: str | None = None


class FilterResult(BaseToolResponse):
    """Filter operation response."""

    session_id: str
    rows_before: int
    rows_after: int
    rows_filtered: int
    conditions_applied: int


class SortResult(BaseToolResponse):
    """Sort operation response."""

    session_id: str
    sorted_by: list[str]
    ascending: list[bool]
    rows_affected: int


class ColumnTransformResult(BaseToolResponse):
    """Column transformation response."""

    session_id: str
    column: str
    operation: str
    rows_affected: int
    original_sample: list[CsvCellValue] | None = None
    transformed_sample: list[CsvCellValue] | None = None


class DuplicateRemovalResult(BaseToolResponse):
    """Duplicate removal response."""

    session_id: str
    rows_before: int
    rows_after: int
    duplicates_removed: int
    subset_columns: list[str] | None = None
    keep_strategy: str


class FillMissingResult(BaseToolResponse):
    """Fill missing values response."""

    session_id: str
    strategy: str
    columns_affected: list[str]
    nulls_before: int
    nulls_after: int
    fill_value: CsvCellValue = None


class StringOperationResult(BaseToolResponse):
    """String operation response."""

    session_id: str
    column: str
    operation: str
    rows_affected: int
    sample_before: list[str] | None = None
    sample_after: list[str] | None = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


# Implementation: Session retrieval with validation and error handling
def _get_session_data(session_id: str) -> tuple[Any, pd.DataFrame]:
    """Get session and DataFrame with validation."""
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if not session:
        raise SessionNotFoundError(session_id)
    if not session.has_data():
        raise NoDataLoadedError(session_id)

    df = session.df
    if df is None:  # Type guard since has_data() was checked
        raise NoDataLoadedError(session_id)
    return session, df


# ============================================================================
# FILTER AND SORT OPERATIONS
# ============================================================================


# Implementation: Multi-condition filtering with logical operators (==, !=, >, <, contains, etc.)
async def filter_rows_with_pydantic(
    session_id: str,
    conditions: list[dict[str, Any]],
    mode: Literal["and", "or"] = "and",
) -> FilterResult:
    """Filter DataFrame rows with multiple conditions."""
    try:
        session, df = _get_session_data(session_id)
        rows_before = len(df)

        # Initialize mask based on mode
        mask = pd.Series([mode == "and"] * len(df))

        for condition in conditions:
            column = condition.get("column")
            operator = condition.get("operator")
            value = condition.get("value")

            if column is None or column not in df.columns:
                raise ColumnNotFoundError(str(column or "None"), df.columns.tolist())

            col_data = df[column]

            # Apply operator
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
            elif operator == "not_null":
                condition_mask = col_data.notna()
            else:
                raise InvalidParameterError(
                    "operator",
                    operator,
                    "Valid operators: ==, !=, >, <, >=, <=, contains, starts_with, ends_with, in, not_in, is_null, not_null",
                )

            mask = mask & condition_mask if mode == "and" else mask | condition_mask

        # Apply filter
        session.df = df[mask].reset_index(drop=True)
        rows_after = len(session.df)

        # Record operation
        session.record_operation(
            OperationType.FILTER,
            {
                "conditions": conditions,
                "mode": mode,
                "rows_before": rows_before,
                "rows_after": rows_after,
            },
        )

        return FilterResult(
            session_id=session_id,
            rows_before=rows_before,
            rows_after=rows_after,
            rows_filtered=rows_before - rows_after,
            conditions_applied=len(conditions),
        )

    except (
        SessionNotFoundError,
        NoDataLoadedError,
        ColumnNotFoundError,
        InvalidParameterError,
    ) as e:
        logger.error(f"Filter operation failed: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Unexpected error in filter_rows: {e}")
        raise ToolError(f"Filter operation failed: {e!s}") from e


# Implementation: Multi-column sorting with flexible column specification and sort order
async def sort_data_with_pydantic(
    session_id: str,
    columns: list[str] | list[dict[str, Any]],
    ascending: bool | list[bool] = True,
) -> SortResult:
    """Sort DataFrame by one or more columns."""
    try:
        session, df = _get_session_data(session_id)

        # Parse columns parameter - handle both string lists and dict lists
        if columns and isinstance(columns[0], dict):
            # We know columns is list[dict] if first element is dict
            dict_columns = columns  # Type narrowing
            sort_columns = [col["column"] for col in dict_columns if isinstance(col, dict)]
            sort_ascending = [
                col.get("ascending", True) for col in dict_columns if isinstance(col, dict)
            ]
        else:
            # columns is list[str]
            sort_columns = [str(col) for col in columns]
            sort_ascending = (
                ascending if isinstance(ascending, list) else [ascending] * len(columns)
            )

        # Validate columns
        missing_cols = [col for col in sort_columns if col not in df.columns]
        if missing_cols:
            raise ToolError(f"Columns not found: {missing_cols}")

        # Sort data
        session.df = df.sort_values(by=sort_columns, ascending=sort_ascending).reset_index(
            drop=True
        )

        # Record operation
        session.record_operation(
            OperationType.SORT,
            {"columns": sort_columns, "ascending": sort_ascending},
        )

        return SortResult(
            session_id=session_id,
            sorted_by=sort_columns,
            ascending=sort_ascending,
            rows_affected=len(df),
        )

    except (SessionNotFoundError, NoDataLoadedError) as e:
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Sort operation failed: {e}")
        raise ToolError(f"Sort operation failed: {e!s}") from e


# Implementation: Duplicate row removal with column subset and keep strategy options
async def remove_duplicates_with_pydantic(
    session_id: str,
    subset: list[str] | None = None,
    keep: Literal["first", "last", False] = "first",
) -> DuplicateRemovalResult:
    """Remove duplicate rows from DataFrame."""
    try:
        session, df = _get_session_data(session_id)
        rows_before = len(df)

        if subset:
            missing_cols = [col for col in subset if col not in df.columns]
            if missing_cols:
                raise ToolError(f"Columns not found: {missing_cols}")

        # Remove duplicates
        session.df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        rows_after = len(session.df)

        # Record operation
        session.record_operation(
            OperationType.REMOVE_DUPLICATES,
            {
                "subset": subset,
                "keep": keep,
                "rows_before": rows_before,
                "rows_after": rows_after,
            },
        )

        return DuplicateRemovalResult(
            session_id=session_id,
            rows_before=rows_before,
            rows_after=rows_after,
            duplicates_removed=rows_before - rows_after,
            subset_columns=subset,
            keep_strategy=str(keep),
        )

    except (SessionNotFoundError, NoDataLoadedError) as e:
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Remove duplicates failed: {e}")
        raise ToolError(f"Failed to remove duplicates: {e!s}") from e


# ============================================================================
# COLUMN TRANSFORMATION OPERATIONS
# ============================================================================


# Implementation: Missing value handling with multiple strategies (drop, fill, forward, backward, mean, median, mode)
async def fill_missing_values_with_pydantic(
    session_id: str,
    columns: list[str] | None = None,
    strategy: Literal["drop", "fill", "forward", "backward", "mean", "median", "mode"] = "drop",
    value: CsvCellValue = None,
) -> FillMissingResult:
    """Fill or remove missing values in columns."""
    try:
        session, df = _get_session_data(session_id)

        # Determine target columns
        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ToolError(f"Columns not found: {missing_cols}")
            target_cols = columns
        else:
            target_cols = df.columns.tolist()

        # Count nulls before
        nulls_before = df[target_cols].isna().sum().sum()

        # Apply strategy
        if strategy == "drop":
            session.df = df.dropna(subset=target_cols)
        elif strategy == "fill":
            if value is None:
                raise ToolError("Value required for 'fill' strategy")
            session.df[target_cols] = df[target_cols].fillna(value)
        elif strategy == "forward":
            session.df[target_cols] = df[target_cols].ffill()
        elif strategy == "backward":
            session.df[target_cols] = df[target_cols].bfill()
        elif strategy == "mean":
            for col in target_cols:
                if df[col].dtype in ["int64", "float64"]:
                    session.df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            for col in target_cols:
                if df[col].dtype in ["int64", "float64"]:
                    session.df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            for col in target_cols:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    session.df[col] = df[col].fillna(mode_val[0])
        else:
            raise ToolError(f"Unknown strategy: {strategy}")

        # Count nulls after
        nulls_after = session.df[target_cols].isna().sum().sum() if strategy != "drop" else 0

        # Record operation
        session.record_operation(
            OperationType.FILL_MISSING,
            {
                "strategy": strategy,
                "value": str(value) if value is not None else None,
                "columns": target_cols,
                "nulls_before": nulls_before,
                "nulls_after": nulls_after,
            },
        )

        return FillMissingResult(
            session_id=session_id,
            strategy=strategy,
            columns_affected=target_cols,
            nulls_before=int(nulls_before),
            nulls_after=int(nulls_after),
            fill_value=value,
        )

    except (SessionNotFoundError, NoDataLoadedError) as e:
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Fill missing values failed: {e}")
        raise ToolError(f"Failed to fill missing values: {e!s}") from e


# Implementation: Text case transformation (upper, lower, title, capitalize)
async def transform_column_case_with_pydantic(
    session_id: str,
    column: str,
    transform: Literal["upper", "lower", "title", "capitalize"],
) -> StringOperationResult:
    """Transform text case in column."""
    try:
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ToolError(f"Column '{column}' not found")

        # Get sample before
        sample_before = df[column].head(5).tolist()

        # Apply transformation
        if transform == "upper":
            session.df[column] = df[column].astype(str).str.upper()
        elif transform == "lower":
            session.df[column] = df[column].astype(str).str.lower()
        elif transform == "title":
            session.df[column] = df[column].astype(str).str.title()
        elif transform == "capitalize":
            session.df[column] = df[column].astype(str).str.capitalize()
        else:
            raise ToolError(f"Unknown transform: {transform}")

        # Get sample after
        sample_after = session.df[column].head(5).tolist()

        # Record operation
        session.record_operation(
            OperationType.UPDATE_COLUMN,
            {"column": column, "transform": transform},
        )

        return StringOperationResult(
            session_id=session_id,
            column=column,
            operation=f"case_{transform}",
            rows_affected=len(df),
            sample_before=[str(v) for v in sample_before],
            sample_after=[str(v) for v in sample_after],
        )

    except (SessionNotFoundError, NoDataLoadedError) as e:
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Transform column case failed: {e}")
        raise ToolError(f"Failed to transform column case: {e!s}") from e


# Implementation: Whitespace removal from string values
async def strip_column_with_pydantic(
    session_id: str,
    column: str,
) -> StringOperationResult:
    """Strip whitespace from column values."""
    try:
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ToolError(f"Column '{column}' not found")

        # Get sample before
        sample_before = df[column].head(5).tolist()

        # Strip whitespace
        session.df[column] = df[column].astype(str).str.strip()

        # Get sample after
        sample_after = session.df[column].head(5).tolist()

        # Record operation
        session.record_operation(
            OperationType.UPDATE_COLUMN,
            {"column": column, "operation": "strip"},
        )

        return StringOperationResult(
            session_id=session_id,
            column=column,
            operation="strip",
            rows_affected=len(df),
            sample_before=[str(v) for v in sample_before],
            sample_after=[str(v) for v in sample_after],
        )

    except (SessionNotFoundError, NoDataLoadedError) as e:
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Strip column failed: {e}")
        raise ToolError(f"Failed to strip column: {e!s}") from e
