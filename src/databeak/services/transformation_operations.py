"""Data transformation operations for CSV manipulation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from fastmcp import Context

import pandas as pd
from fastmcp.exceptions import ToolError

from ..exceptions import (
    ColumnNotFoundError,
    InvalidParameterError,
    NoDataLoadedError,
    SessionNotFoundError,
)
from ..models.csv_session import get_session_manager
from ..models.data_models import OperationType
from ..models.tool_responses import (
    CellValueResult,
    ColumnDataResult,
    ColumnOperationResult,
    DeleteRowResult,
    FilterOperationResult,
    InsertRowResult,
    RowDataResult,
    SetCellResult,
    SortDataResult,
    UpdateRowResult,
)
from ..utils.pydantic_validators import parse_json_string_to_dict
from ..utils.secure_evaluator import _get_secure_evaluator
from ..utils.validators import convert_pandas_na_list

# Type aliases for better type safety (non-conflicting)
CellValue = str | int | float | bool | None
RowData = dict[str, CellValue] | list[CellValue]

# Note: FilterCondition and OperationResult are imported from data_models.py when needed
# To avoid conflicts with tool response models, use them as dict types in function signatures

logger = logging.getLogger(__name__)


# Implementation: Session validation and DataFrame retrieval with error handling
def _get_session_data(session_id: str) -> tuple[Any, pd.DataFrame]:
    """Get validated session and DataFrame."""
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if not session:
        raise SessionNotFoundError(session_id)
    if not session.has_data():
        raise NoDataLoadedError(session_id)

    df = session.df
    if df is None:  # Additional defensive check
        raise NoDataLoadedError(session_id)
    return session, df


# Implementation: Multi-condition row filtering with logical operators and comprehensive comparison support
async def filter_rows(
    session_id: str,
    conditions: list[dict[str, str | CellValue]],
    mode: str = "and",
    ctx: Context | None = None,  # noqa: ARG001
) -> FilterOperationResult:
    """Filter DataFrame rows based on conditions."""
    try:
        session, df = _get_session_data(session_id)
        # Initialize mask based on mode: AND starts True, OR starts False
        mask = pd.Series([mode == "and"] * len(df))

        for condition in conditions:
            column = condition.get("column")
            operator = condition.get("operator")
            value = condition.get("value")

            if column is None or column not in df.columns:
                raise ColumnNotFoundError(str(column or "None"), df.columns.tolist())

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

        session.df = df[mask].reset_index(drop=True)
        session.record_operation(
            OperationType.FILTER,
            {
                "conditions": conditions,
                "mode": mode,
                "rows_before": len(df),
                "rows_after": len(session.df),
            },
        )

        return FilterOperationResult(
            session_id=session_id,
            rows_before=len(df),
            rows_after=len(session.df),
            rows_filtered=len(df) - len(session.df),
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
    except (ValueError, TypeError) as e:
        logger.error(f"Data type error in filter operation: {e!s}")
        raise ToolError(f"Data type error: {e}") from e
    except pd.errors.ParserError as e:
        logger.error(f"Pandas parsing error: {e!s}")
        raise ToolError(f"Parsing error: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in filter operation: {e!s}")
        raise ToolError(f"Unexpected error: {e}") from e


# Implementation: Multi-column DataFrame sorting with flexible column specification
async def sort_data(
    session_id: str,
    columns: list[str | dict[str, str]],
    ctx: Context | None = None,  # noqa: ARG001
) -> SortDataResult:
    """Sort DataFrame by one or more columns."""
    try:
        session, df = _get_session_data(session_id)

        # Parse columns into names and ascending flags
        sort_columns: list[str] = []
        ascending: list[bool] = []

        for col in columns:
            if isinstance(col, str):
                sort_columns.append(col)
                ascending.append(True)
            elif isinstance(col, dict):
                sort_columns.append(col["column"])
                ascending.append(bool(col.get("ascending", True)))
            else:
                raise ToolError(f"Invalid column specification: {col}")

        # Validate columns exist
        for col in sort_columns:
            if col not in df.columns:
                raise ToolError(f"Column '{col}' not found")

        sorted_df = df.sort_values(by=sort_columns, ascending=ascending).reset_index(drop=True)
        session.df = sorted_df
        session.record_operation(
            OperationType.SORT, {"columns": sort_columns, "ascending": ascending}
        )

        return SortDataResult(
            session_id=session_id,
            sorted_by=sort_columns,
            ascending=ascending,
            rows_processed=len(sorted_df),
        )

    except Exception as e:
        logger.error(f"Error sorting data: {e!s}")
        raise ToolError(f"Failed to sort data: {e}") from e


async def select_columns(
    session_id: str,
    columns: list[str],
    ctx: Context | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    """Select specific columns from the dataframe.

    Args:
        session_id: Session identifier
        columns: List of column names to keep
        ctx: FastMCP context

    Returns:
        Dict with success status and selected columns
    """
    try:
        session, df = _get_session_data(session_id)

        # Validate columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ToolError(f"Columns not found: {missing_cols}")

        # Track counts before modification
        columns_before = len(df.columns)

        session.df = df[columns].copy()
        session.record_operation(
            OperationType.SELECT,
            {
                "columns": columns,
                "columns_before": df.columns.tolist(),
                "columns_after": columns,
            },
        )

        return {
            "session_id": session_id,
            "selected_columns": columns,
            "columns_before": columns_before,
            "columns_after": len(columns),
        }

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
        mapping: Dict mapping old column names to new names
        ctx: FastMCP context

    Returns:
        Dict with success status and renamed columns
    """
    try:
        session, df = _get_session_data(session_id)

        # Validate columns exist
        missing_cols = [col for col in mapping if col not in df.columns]
        if missing_cols:
            raise ToolError(f"Columns not found: {missing_cols}")

        session.df = df.rename(columns=mapping)
        session.record_operation(OperationType.RENAME, {"mapping": mapping})

        return {
            "session_id": session_id,
            "renamed": mapping,
            "columns": session.df.columns.tolist(),
        }

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
        value: Default value for all rows (scalar or list)
        formula: Python expression to calculate values (e.g., "col1 + col2")
        ctx: FastMCP context

    Returns:
        Dict with success status
    """
    try:
        session, df = _get_session_data(session_id)

        if name in df.columns:
            raise ToolError(f"Column '{name}' already exists")

        if formula:
            # Evaluate formula using secure expression evaluator
            try:
                evaluator = _get_secure_evaluator()
                result = evaluator.evaluate_simple_formula(formula, df)
                session.df[name] = result
            except Exception as e:
                raise ToolError(f"Formula evaluation failed: {e}") from e
        elif isinstance(value, list):
            if len(value) != len(df):
                raise ToolError(
                    f"Value list length ({len(value)}) doesn't match row count ({len(df)})"
                )
            session.df[name] = value
        else:
            # Scalar value or None
            session.df[name] = value

        session.record_operation(
            OperationType.ADD_COLUMN,
            {
                "name": name,
                "value": str(value) if value is not None else None,
                "formula": formula,
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="add_column",
            rows_affected=len(session.df),
            columns_affected=[name],
        )

    except Exception as e:
        logger.error(f"Error adding column: {e!s}")
        raise ToolError(f"Error: {e}") from e


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
        Dict with success status and removed columns
    """
    try:
        session, df = _get_session_data(session_id)

        # Validate columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ToolError(f"Columns not found: {missing_cols}")

        session.df = df.drop(columns=columns)
        session.record_operation(OperationType.REMOVE_COLUMN, {"columns": columns})

        return ColumnOperationResult(
            session_id=session_id,
            operation="remove_columns",
            rows_affected=len(session.df),
            columns_affected=columns,
        )

    except Exception as e:
        logger.error(f"Error removing columns: {e!s}")
        raise ToolError(f"Error: {e}") from e


async def change_column_type(
    session_id: str,
    column: str,
    dtype: str,
    errors: Literal["raise", "coerce"] = "coerce",
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Change the data type of a column.

    Args:
        session_id: Session identifier
        column: Column name to change
        dtype: Target data type ('int', 'float', 'str', 'bool', 'datetime', 'category')
        errors: How to handle conversion errors ('raise', 'coerce', 'ignore')
        ctx: FastMCP context

    Returns:
        Dict with success status and conversion info
    """
    try:
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ToolError(f"Column '{column}' not found")

        original_dtype = str(df[column].dtype)

        # Convert based on target dtype
        if dtype == "int":
            session.df[column] = pd.to_numeric(df[column], errors=errors).astype("Int64")
        elif dtype == "float":
            session.df[column] = pd.to_numeric(df[column], errors=errors)
        elif dtype == "str":
            session.df[column] = df[column].astype(str)
        elif dtype == "bool":
            session.df[column] = df[column].astype(bool)
        elif dtype == "datetime":
            session.df[column] = pd.to_datetime(df[column], errors=errors)
        elif dtype == "category":
            session.df[column] = df[column].astype("category")
        else:
            raise ToolError(f"Unsupported dtype: {dtype}")

        session.record_operation(
            OperationType.CHANGE_TYPE,
            {
                "column": column,
                "from_type": original_dtype,
                "to_type": dtype,
                "errors": errors,
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="change_column_type",
            rows_affected=len(session.df),
            columns_affected=[column],
        )

    except Exception as e:
        logger.error(f"Error changing column type: {e!s}")
        raise ToolError(f"Error: {e}") from e


async def fill_missing_values(
    session_id: str,
    strategy: str = "drop",
    value: CellValue = None,
    columns: list[str] | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Fill or remove missing values.

    Args:
        session_id: Session identifier
        strategy: One of 'drop', 'fill', 'forward', 'backward', 'mean', 'median', 'mode'
        value: Value to fill with (for 'fill' strategy)
        columns: Specific columns to apply to (None for all)
        ctx: FastMCP context

    Returns:
        Dict with success status and fill info
    """
    try:
        session, df = _get_session_data(session_id)

        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ToolError(f"Columns not found: {missing_cols}")
            target_cols = columns
        else:
            target_cols = df.columns.tolist()

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

        session.record_operation(
            OperationType.FILL_MISSING,
            {
                "strategy": strategy,
                "value": str(value) if value is not None else None,
                "columns": target_cols,
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="fill_missing_values",
            rows_affected=len(session.df),
            columns_affected=target_cols,
        )

    except Exception as e:
        logger.error(f"Error filling missing values: {e!s}")
        raise ToolError(f"Error: {e}") from e


async def update_column(
    session_id: str,
    column: str,
    operation: str,
    value: CellValue = None,
    pattern: str | None = None,
    replacement: str | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Update values in a specific column with simple operations.

    Args:
        session_id: Session identifier
        column: Column name to update
        operation: Operation type - 'replace', 'extract', 'split', 'strip', 'upper', 'lower', 'fill'
        value: Value for certain operations (e.g., fill value)
        pattern: Pattern for replace/extract operations (regex supported)
        replacement: Replacement string for replace operation
        ctx: FastMCP context

    Returns:
        Dict with success status and update info
    """
    try:
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ToolError(f"Column '{column}' not found")

        original_values_sample = convert_pandas_na_list(df[column].head(5).tolist())

        if operation == "replace":
            if pattern is None or replacement is None:
                raise ToolError("Pattern and replacement required for replace operation")
            session.df[column] = (
                df[column].astype(str).str.replace(pattern, replacement, regex=True)
            )

        elif operation == "extract":
            if pattern is None:
                raise ToolError("Pattern required for extract operation")
            session.df[column] = df[column].astype(str).str.extract(pattern, expand=False)

        elif operation == "split":
            if pattern is None:
                pattern = " "
            if value is not None and isinstance(value, int):
                # Extract specific part after split
                session.df[column] = df[column].astype(str).str.split(pattern).str[value]
            else:
                # Just do the split, take first part
                session.df[column] = df[column].astype(str).str.split(pattern).str[0]

        elif operation == "strip":
            session.df[column] = df[column].astype(str).str.strip()

        elif operation == "upper":
            session.df[column] = df[column].astype(str).str.upper()

        elif operation == "lower":
            session.df[column] = df[column].astype(str).str.lower()

        elif operation == "fill":
            if value is None:
                raise ToolError("Value required for fill operation")
            session.df[column] = df[column].fillna(value)

        else:
            raise ToolError(f"Unknown operation: {operation}")

        updated_values_sample = convert_pandas_na_list(session.df[column].head(5).tolist())

        session.record_operation(
            OperationType.UPDATE_COLUMN,
            {
                "column": column,
                "operation": operation,
                "pattern": pattern,
                "replacement": replacement,
                "value": str(value) if value is not None else None,
            },
        )

        return ColumnOperationResult(
            success=True,
            session_id=session_id,
            operation=f"update_column_{operation}",
            rows_affected=len(session.df),
            columns_affected=[column],
            original_sample=original_values_sample[:5] if original_values_sample else [],
            updated_sample=updated_values_sample[:5] if updated_values_sample else [],
        )

    except Exception as e:
        logger.error(f"Error updating column: {e!s}")
        raise ToolError(f"Error updating column: {e}") from e


async def remove_duplicates(
    session_id: str,
    subset: list[str] | None = None,
    keep: Literal["first", "last", "none"] = "first",
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Remove duplicate rows.

    Args:
        session_id: Session identifier
        subset: Column names to consider for duplicates (None for all)
        keep: Which duplicates to keep ('first', 'last', False to drop all)
        ctx: FastMCP context

    Returns:
        Dict with success status and duplicate info
    """
    try:
        session, df = _get_session_data(session_id)
        rows_before = len(df)

        if subset:
            missing_cols = [col for col in subset if col not in df.columns]
            if missing_cols:
                raise ToolError(f"Columns not found: {missing_cols}")

        # Convert keep parameter
        keep_param: Literal["first", "last"] | Literal[False] = keep if keep != "none" else False

        session.df = df.drop_duplicates(subset=subset, keep=keep_param).reset_index(drop=True)
        rows_after = len(session.df)

        session.record_operation(
            OperationType.REMOVE_DUPLICATES,
            {"subset": subset, "keep": keep, "rows_removed": rows_before - rows_after},
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="remove_duplicates",
            rows_affected=rows_after,
            columns_affected=subset if subset else df.columns.tolist(),
        )

    except Exception as e:
        logger.error(f"Error removing duplicates: {e!s}")
        raise ToolError(f"Error: {e}") from e


# ============================================================================
# CELL-LEVEL ACCESS METHODS
# ============================================================================


async def get_cell_value(
    session_id: str,
    row_index: int,
    column: str | int,
    ctx: Context | None = None,  # noqa: ARG001
) -> CellValueResult:
    """Get the value of a specific cell.

    Args:
        session_id: Session identifier
        row_index: Row index (0-based)
        column: Column name (str) or column index (int, 0-based)
        ctx: FastMCP context

    Returns:
        Dict with success status and cell value

    Example:
        get_cell_value("session123", 0, "name") -> {"success": True, "value": "John", "coordinates": {"row": 0, "column": "name"}}
        get_cell_value("session123", 2, 1) -> {"success": True, "value": 25, "coordinates": {"row": 2, "column": "age"}}
    """
    try:
        session, df = _get_session_data(session_id)

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
                raise ToolError(f"Column '{column}' not found")
            column_name = column

        # Get the cell value
        cell_value = df.iloc[row_index][column_name]

        # Handle pandas/numpy types for JSON serialization
        if pd.isna(cell_value):
            cell_value = None
        elif hasattr(cell_value, "item"):
            cell_value = cell_value.item()

        return CellValueResult(
            value=cell_value,
            coordinates={"row": row_index, "column": column_name},
            data_type=str(df.dtypes[column_name]),
        )

    except Exception as e:
        logger.error(f"Error getting cell value: {e!s}")
        raise ToolError(f"Error: {e}") from e


async def set_cell_value(
    session_id: str,
    row_index: int,
    column: str | int,
    value: CellValue,
    ctx: Context | None = None,  # noqa: ARG001
) -> SetCellResult:
    """Set the value of a specific cell.

    Args:
        session_id: Session identifier
        row_index: Row index (0-based)
        column: Column name (str) or column index (int, 0-based)
        value: New value for the cell
        ctx: FastMCP context

    Returns:
        Dict with success status and update info

    Example:
        set_cell_value("session123", 0, "name", "Jane") -> {"success": True, "old_value": "John", "new_value": "Jane"}
    """
    try:
        session, df = _get_session_data(session_id)

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
                raise ToolError(f"Column '{column}' not found")
            column_name = column

        # Get old value
        old_value = df.iloc[row_index][column_name]
        if pd.isna(old_value):
            old_value = None
        elif hasattr(old_value, "item"):
            old_value = old_value.item()

        # Set new value
        session.df.iloc[row_index, session.df.columns.get_loc(column_name)] = value

        # Record operation
        session.record_operation(
            OperationType.UPDATE_COLUMN,  # Reuse existing operation type
            {
                "operation": "set_cell",
                "coordinates": {"row": row_index, "column": column_name},
                "old_value": str(old_value) if old_value is not None else None,
                "new_value": str(value) if value is not None else None,
            },
        )

        return SetCellResult(
            coordinates={"row": row_index, "column": column_name},
            old_value=old_value,
            new_value=value,
            data_type=str(df.dtypes[column_name]),
        )

    except Exception as e:
        logger.error(f"Error setting cell value: {e!s}")
        raise ToolError(f"Error: {e}") from e


async def get_row_data(
    session_id: str,
    row_index: int,
    columns: list[str] | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> RowDataResult:
    """Get data from a specific row.

    Args:
        session_id: Session identifier
        row_index: Row index (0-based)
        columns: Optional list of column names to include (None for all columns)
        ctx: FastMCP context

    Returns:
        Dict with success status and row data

    Example:
        get_row_data("session123", 0) -> {"success": True, "data": {"name": "John", "age": 30}, "row_index": 0}
        get_row_data("session123", 1, ["name", "age"]) -> {"success": True, "data": {"name": "Jane", "age": 25}}
    """
    try:
        session, df = _get_session_data(session_id)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            raise ToolError(f"Row index {row_index} out of range (0-{len(df) - 1})")

        # Get row data
        if columns is None:
            row_data = df.iloc[row_index].to_dict()
        else:
            # Validate columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ToolError(f"Columns not found: {missing_cols}")

            row_data = df.iloc[row_index][columns].to_dict()

        # Handle pandas/numpy types for JSON serialization
        for key, value in row_data.items():
            if pd.isna(value):
                row_data[key] = None
            elif hasattr(value, "item"):
                row_data[key] = value.item()

        return RowDataResult(
            session_id=session_id,
            row_index=row_index,
            data=row_data,
            columns=list(row_data.keys()),
        )

    except Exception as e:
        logger.error(f"Error getting row data: {e!s}")
        raise ToolError(f"Error: {e}") from e


async def get_column_data(
    session_id: str,
    column: str,
    start_row: int | None = None,
    end_row: int | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnDataResult:
    """Get data from a specific column, optionally sliced by row range.

    Args:
        session_id: Session identifier
        column: Column name
        start_row: Starting row index (0-based, inclusive). None for beginning
        end_row: Ending row index (0-based, exclusive). None for end
        ctx: FastMCP context

    Returns:
        Dict with success status and column data

    Example:
        get_column_data("session123", "age") -> {"success": True, "data": [30, 25, 35], "column": "age"}
        get_column_data("session123", "name", 0, 2) -> {"success": True, "data": ["John", "Jane"]}
    """
    try:
        session, df = _get_session_data(session_id)

        # Validate column exists
        if column not in df.columns:
            raise ToolError(f"Column '{column}' not found")

        # Validate and set row range
        total_rows = len(df)
        if start_row is None:
            start_row = 0
        if end_row is None:
            end_row = total_rows

        if start_row < 0 or start_row >= total_rows:
            raise ToolError(f"Start row {start_row} out of range (0-{total_rows - 1})")

        if end_row < start_row or end_row > total_rows:
            raise ToolError(f"End row {end_row} invalid (must be > start_row and <= {total_rows})")

        # Get column data slice
        column_data = df[column].iloc[start_row:end_row].tolist()

        # Handle pandas/numpy types for JSON serialization
        for i, value in enumerate(column_data):
            if pd.isna(value):
                column_data[i] = None
            elif hasattr(value, "item"):
                column_data[i] = value.item()

        return ColumnDataResult(
            session_id=session_id,
            column=column,
            values=column_data,
            total_values=len(column_data),
            start_row=start_row,
            end_row=end_row,
        )

    except Exception as e:
        logger.error(f"Error getting column data: {e!s}")
        raise ToolError(f"Error: {e}") from e


# ============================================================================
# FOCUSED COLUMN UPDATE METHODS (Replacing operation-parameter pattern)
# ============================================================================


async def replace_in_column(
    session_id: str,
    column: str,
    pattern: str,
    replacement: str,
    regex: bool = True,
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Replace patterns in a column with replacement text.

    Args:
        session_id: Session identifier
        column: Column name to update
        pattern: Pattern to search for (regex or literal string)
        replacement: Replacement string
        regex: Whether to treat pattern as regex (default: True)
        ctx: FastMCP context

    Returns:
        Dict with success status and replacement info

    Example:
        replace_in_column("session123", "name", "Mr\\.", "Mister") -> Replace "Mr." with "Mister"
    """
    try:
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ToolError(f"Column '{column}' not found")

        # Get original sample
        original_sample = convert_pandas_na_list(df[column].head(5).tolist())

        # Perform replacement
        session.df[column] = df[column].astype(str).str.replace(pattern, replacement, regex=regex)

        # Get updated sample
        updated_sample = convert_pandas_na_list(session.df[column].head(5).tolist())

        session.record_operation(
            OperationType.UPDATE_COLUMN,
            {
                "operation": "replace",
                "column": column,
                "pattern": pattern,
                "replacement": replacement,
                "regex": regex,
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="replace_in_column",
            rows_affected=len(session.df),
            columns_affected=[column],
            original_sample=original_sample,
            updated_sample=updated_sample,
        )

    except Exception as e:
        logger.error(f"Error replacing in column: {e!s}")
        raise ToolError(f"Error: {e}") from e


async def extract_from_column(
    session_id: str,
    column: str,
    pattern: str,
    expand: bool = False,
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Extract patterns from a column using regex.

    Args:
        session_id: Session identifier
        column: Column name to update
        pattern: Regex pattern to extract (use capturing groups)
        expand: Whether to expand to multiple columns if multiple groups
        ctx: FastMCP context

    Returns:
        Dict with success status and extraction info

    Example:
        extract_from_column("session123", "email", r"(.+)@(.+)") -> Extract username and domain
    """
    try:
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ToolError(f"Column '{column}' not found")

        # Perform extraction
        session.df[column] = df[column].astype(str).str.extract(pattern, expand=False)

        session.record_operation(
            OperationType.UPDATE_COLUMN,
            {
                "operation": "extract",
                "column": column,
                "pattern": pattern,
                "expand": expand,
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="extract_from_column",
            rows_affected=len(session.df),
            columns_affected=[column],
        )

    except Exception as e:
        logger.error(f"Error extracting from column: {e!s}")
        raise ToolError(f"Error: {e}") from e


async def split_column(
    session_id: str,
    column: str,
    delimiter: str = " ",
    part_index: int | None = None,
    expand_to_columns: bool = False,
    new_columns: list[str] | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Split column values by delimiter.

    Args:
        session_id: Session identifier
        column: Column name to update
        delimiter: String to split on (default: space)
        part_index: Which part to keep (0-based index). None keeps first part
        expand_to_columns: Whether to expand splits into multiple columns
        ctx: FastMCP context

    Returns:
        Dict with success status and split info

    Example:
        split_column("session123", "name", " ", 0) -> Keep first part of name
        split_column("session123", "full_name", " ", expand_to_columns=True) -> Split into multiple columns
    """
    try:
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ToolError(f"Column '{column}' not found")

        # Perform split
        if expand_to_columns or new_columns:
            # Split into multiple columns
            split_data = df[column].astype(str).str.split(delimiter, expand=True)
            # Replace original column with split columns
            column_names = []
            for i, col_data in enumerate(split_data.columns):
                if new_columns and i < len(new_columns):
                    new_col_name = new_columns[i]
                else:
                    new_col_name = f"{column}_{i}"
                session.df[new_col_name] = split_data[col_data]
                column_names.append(new_col_name)
            # Drop original column
            session.df = session.df.drop(columns=[column])
            columns_affected = column_names
        else:
            # Keep specific part
            if part_index is not None:
                session.df[column] = df[column].astype(str).str.split(delimiter).str[part_index]
            else:
                # Keep first part by default
                session.df[column] = df[column].astype(str).str.split(delimiter).str[0]
            columns_affected = [column]

        session.record_operation(
            OperationType.UPDATE_COLUMN,
            {
                "operation": "split",
                "column": column,
                "delimiter": delimiter,
                "part_index": part_index,
                "expand_to_columns": expand_to_columns,
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="split_column",
            rows_affected=len(session.df),
            columns_affected=columns_affected,
            part_index=part_index,
        )

    except Exception as e:
        logger.error(f"Error splitting column: {e!s}")
        raise ToolError(f"Error: {e}") from e


async def transform_column_case(
    session_id: str,
    column: str,
    transform: Literal["upper", "lower", "title", "capitalize"],
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Transform the case of text in a column.

    Args:
        session_id: Session identifier
        column: Column name to update
        transform: Type of case transformation
        ctx: FastMCP context

    Returns:
        Dict with success status and transformation info

    Example:
        transform_column_case("session123", "name", "title") -> "john doe" becomes "John Doe"
    """
    try:
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ToolError(f"Column '{column}' not found")

        # Perform case transformation
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

        session.record_operation(
            OperationType.UPDATE_COLUMN,
            {
                "operation": "transform_case",
                "column": column,
                "transform": transform,
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="transform_column_case",
            rows_affected=len(session.df),
            columns_affected=[column],
            transform=transform,
        )

    except Exception as e:
        logger.error(f"Error transforming column case: {e!s}")
        raise ToolError(f"Error: {e}") from e


async def strip_column(
    session_id: str,
    column: str,
    chars: str | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Strip whitespace or specified characters from column values.

    Args:
        session_id: Session identifier
        column: Column name to update
        chars: Characters to strip (None for whitespace)
        ctx: FastMCP context

    Returns:
        Dict with success status and strip info

    Example:
        strip_column("session123", "name") -> Remove leading/trailing whitespace
        strip_column("session123", "code", "()") -> Remove parentheses from ends
    """
    try:
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ToolError(f"Column '{column}' not found")

        original_values_sample = convert_pandas_na_list(df[column].head(5).tolist())

        # Perform strip
        if chars is None:
            session.df[column] = df[column].astype(str).str.strip()
        else:
            session.df[column] = df[column].astype(str).str.strip(chars)

        updated_values_sample = convert_pandas_na_list(session.df[column].head(5).tolist())

        session.record_operation(
            OperationType.UPDATE_COLUMN,
            {
                "operation": "strip",
                "column": column,
                "chars": chars,
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="strip_column",
            rows_affected=len(session.df),
            columns_affected=[column],
            original_sample=original_values_sample,
            updated_sample=updated_values_sample,
        )

    except Exception as e:
        logger.error(f"Error stripping column: {e!s}")
        raise ToolError(f"Error: {e}") from e


async def fill_column_nulls(
    session_id: str,
    column: str,
    value: CellValue,
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnOperationResult:
    """Fill null/NaN values in a column with a specified value.

    Args:
        session_id: Session identifier
        column: Column name to update
        value: Value to use for filling nulls
        ctx: FastMCP context

    Returns:
        Dict with success status and fill info

    Example:
        fill_column_nulls("session123", "age", 0) -> Replace NaN ages with 0
        fill_column_nulls("session123", "name", "Unknown") -> Replace missing names with "Unknown"
    """
    try:
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ToolError(f"Column '{column}' not found")

        # Count null values before
        nulls_before = df[column].isna().sum()

        # Fill nulls
        session.df[column] = df[column].fillna(value)

        # Count nulls after
        nulls_after = session.df[column].isna().sum()

        session.record_operation(
            OperationType.UPDATE_COLUMN,
            {
                "operation": "fill_nulls",
                "column": column,
                "value": str(value),
                "nulls_filled": nulls_before - nulls_after,
            },
        )

        return ColumnOperationResult(
            session_id=session_id,
            operation="fill_column_nulls",
            rows_affected=len(session.df),
            columns_affected=[column],
            nulls_filled=int(nulls_before - nulls_after),
        )

    except Exception as e:
        logger.error(f"Error filling column nulls: {e!s}")
        raise ToolError(f"Error: {e}") from e


# ============================================================================
# ROW MANIPULATION METHODS
# ============================================================================


async def insert_row(
    session_id: str,
    row_index: int,
    data: RowData | str,  # Accept string for Claude Code compatibility
    ctx: Context | None = None,  # noqa: ARG001
) -> InsertRowResult:
    """Insert a new row at the specified index.

    Args:
        session_id: Session identifier
        row_index: Index where to insert the row (0-based). Use -1 to append at end
        data: Row data as dict (column_name: value), list of values, or JSON string
              Supports null/None values - JSON null becomes Python None
        ctx: FastMCP context

    Returns:
        Dict with success status and insertion info

    Example:
        insert_row("session123", 1, {"name": "Alice", "age": 28, "city": "Boston"})
        insert_row("session123", -1, ["David", 40, "Miami"])  # Append at end
        insert_row("session123", 0, '{"name": "John", "age": null}')  # JSON string from Claude Code
    """
    try:
        # Handle Claude Code's JSON string serialization issue
        if isinstance(data, str):
            try:
                data = parse_json_string_to_dict(data)
            except ValueError as e:
                raise ToolError(f"Invalid JSON string in data parameter: {e}") from e

        session, df = _get_session_data(session_id)
        rows_before = len(df)

        # Handle append case
        if row_index == -1:
            row_index = len(df)

        # Validate row index
        if row_index < 0 or row_index > len(df):
            raise ToolError(f"Row index {row_index} out of range (0-{len(df)} for insertion)")

        # Prepare row data
        if isinstance(data, dict):
            # Ensure all columns are present
            missing_cols = [col for col in df.columns if col not in data]
            if missing_cols:
                # Fill missing columns with None
                for col in missing_cols:
                    data[col] = None

            # Reorder according to dataframe columns
            row_data = [data.get(col, None) for col in df.columns]
        elif isinstance(data, list):
            if len(data) != len(df.columns):
                raise ToolError(
                    f"Row data length ({len(data)}) must match number of columns ({len(df.columns)})"
                )
            row_data = data
        else:
            raise ToolError("Data must be a dict or list")

        # Create new row as DataFrame
        new_row_df = pd.DataFrame([row_data], columns=df.columns)

        # Insert the row
        if row_index == 0:
            session.df = pd.concat([new_row_df, df], ignore_index=True)
        elif row_index >= len(df):
            session.df = pd.concat([df, new_row_df], ignore_index=True)
        else:
            # Split and insert
            before = df.iloc[:row_index]
            after = df.iloc[row_index:]
            session.df = pd.concat([before, new_row_df, after], ignore_index=True)

        session.record_operation(
            OperationType.UPDATE_COLUMN,  # Reuse existing type
            {
                "operation": "insert_row",
                "row_index": row_index,
                "data": row_data,
                "rows_before": rows_before,
                "rows_after": len(session.df),
            },
        )

        # Convert row_data to dict format expected by InsertRowResult
        if isinstance(row_data, list):
            data_inserted = dict(zip(df.columns, row_data))
        else:
            data_inserted = row_data

        return InsertRowResult(
            session_id=session_id,
            row_index=row_index,
            rows_before=rows_before,
            rows_after=len(session.df),
            data_inserted=data_inserted,
            columns=df.columns.tolist(),
        )

    except Exception as e:
        logger.error(f"Error inserting row: {e!s}")
        raise ToolError(f"Error: {e}") from e


async def delete_row(
    session_id: str,
    row_index: int,
    ctx: Context | None = None,  # noqa: ARG001
) -> DeleteRowResult:
    """Delete a row at the specified index.

    Args:
        session_id: Session identifier
        row_index: Row index to delete (0-based)
        ctx: FastMCP context

    Returns:
        Dict with success status and deletion info

    Example:
        delete_row("session123", 1) -> Delete second row
    """
    try:
        session, df = _get_session_data(session_id)
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
            elif hasattr(value, "item"):
                deleted_data[key] = value.item()

        # Delete the row
        session.df = df.drop(df.index[row_index]).reset_index(drop=True)

        session.record_operation(
            OperationType.UPDATE_COLUMN,  # Reuse existing type
            {
                "operation": "delete_row",
                "row_index": row_index,
                "deleted_data": deleted_data,
                "rows_before": rows_before,
                "rows_after": len(session.df),
            },
        )

        return DeleteRowResult(
            session_id=session_id,
            row_index=row_index,
            rows_before=rows_before,
            rows_after=len(session.df),
            deleted_data=deleted_data,
        )

    except Exception as e:
        logger.error(f"Error deleting row: {e!s}")
        raise ToolError(f"Error: {e}") from e


async def update_row(
    session_id: str,
    row_index: int,
    data: dict[str, CellValue] | str,  # Accept string for Claude Code compatibility
    ctx: Context | None = None,  # noqa: ARG001
) -> UpdateRowResult:
    """Update specific columns in a row with new values.

    Args:
        session_id: Session identifier
        row_index: Row index to update (0-based)
        data: Dict with column names and new values (partial updates allowed) or JSON string
              Supports null/None values - JSON null becomes Python None
        ctx: FastMCP context

    Returns:
        Dict with success status and update info

    Example:
        update_row("session123", 0, {"age": 31, "city": "Boston"}) -> Update age and city for first row
        update_row("session123", 1, '{"phone": null, "email": null}') -> JSON string from Claude Code
    """
    try:
        # Handle Claude Code's JSON string serialization issue
        if isinstance(data, str):
            try:
                data = parse_json_string_to_dict(data)
            except ValueError as e:
                raise ToolError(f"Invalid JSON string in data parameter: {e}") from e

        session, df = _get_session_data(session_id)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            raise ToolError(f"Row index {row_index} out of range (0-{len(df) - 1})")

        # Ensure data is a dict at this point
        if not isinstance(data, dict):
            raise ToolError("Data must be a dictionary after JSON parsing")

        # Validate columns exist
        invalid_cols = [col for col in data if col not in df.columns]
        if invalid_cols:
            raise ToolError(f"Columns not found: {invalid_cols}")

        # Get old values for tracking
        old_values: dict[str, Any] = {}
        for col in data:
            old_val = df.iloc[row_index][col]
            if pd.isna(old_val):
                old_values[col] = None
            elif hasattr(old_val, "item"):
                old_values[col] = old_val.item()
            else:
                old_values[col] = old_val

        # Update the row
        for column, value in data.items():
            session.df.iloc[row_index, session.df.columns.get_loc(column)] = value

        # Get new values for tracking
        new_values: dict[str, Any] = {}
        for col in data:
            new_val = session.df.iloc[row_index][col]
            if pd.isna(new_val):
                new_values[col] = None
            elif hasattr(new_val, "item"):
                new_values[col] = new_val.item()
            else:
                new_values[col] = new_val

        session.record_operation(
            OperationType.UPDATE_COLUMN,  # Reuse existing type
            {
                "operation": "update_row",
                "row_index": row_index,
                "columns_updated": list(data.keys()),
                "old_values": old_values,
                "new_values": new_values,
            },
        )

        return UpdateRowResult(
            row_index=row_index,
            columns_updated=list(data.keys()),
            old_values=old_values,
            new_values=new_values,
            changes_made=len(data),
        )

    except Exception as e:
        logger.error(f"Error updating row: {e!s}")
        raise ToolError(f"Error: {e}") from e
