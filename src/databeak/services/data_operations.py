"""Core data operations and utilities for CSV data manipulation."""

from __future__ import annotations

import pandas as pd

from ..exceptions import InvalidRowIndexError, NoDataLoadedError, SessionNotFoundError
from ..models import get_session_manager
from ..models.typed_dicts import CellValue, DataPreviewResult, InternalDataSummary


# Implementation: Convert DataFrame to structured preview with row indices and type handling
def create_data_preview_with_indices(df: pd.DataFrame, num_rows: int = 5) -> DataPreviewResult:
    """Create data preview with row indices and metadata."""
    preview_df = df.head(num_rows)

    # Create records with row indices
    preview_records = []
    for _, (row_idx, row) in enumerate(preview_df.iterrows()):
        # Handle pandas index types safely
        row_index_val = row_idx if isinstance(row_idx, int) else 0
        # Convert all keys to strings and handle pandas/numpy types
        record: dict[str, CellValue] = {
            "__row_index__": row_index_val
        }  # Include original row index
        row_dict = row.to_dict()
        for key, value in row_dict.items():
            str_key = str(key)
            if pd.isna(value):
                record[str_key] = None
            elif isinstance(value, pd.Timestamp):
                record[str_key] = str(value)
            elif hasattr(value, "item"):
                record[str_key] = value.item()
            else:
                record[str_key] = value

        preview_records.append(record)

    return DataPreviewResult(
        records=preview_records,
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=df.columns.tolist(),
        preview_rows=len(preview_records),
    )


# Implementation: Comprehensive data analysis including shape, types, memory usage, nulls
def get_data_summary(session_id: str) -> InternalDataSummary:
    """Get comprehensive data summary for session."""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if not session:
        raise SessionNotFoundError(session_id)
    if session.df is None:
        raise NoDataLoadedError(session_id)

    df = session.df

    return InternalDataSummary(
        session_id=session_id,
        shape=df.shape,
        columns=df.columns.tolist(),
        dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
        memory_usage_mb=round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        null_counts=df.isnull().sum().to_dict(),
        preview=create_data_preview_with_indices(df, 10),
    )


# Implementation: Boundary check for DataFrame row access
def validate_row_index(df: pd.DataFrame, row_index: int) -> None:
    """Validate row index is within DataFrame bounds."""
    if row_index < 0 or row_index >= len(df):
        raise InvalidRowIndexError(row_index, len(df) - 1)


# Implementation: Column existence check with error handling
def validate_column_exists(df: pd.DataFrame, column: str) -> None:
    """Validate column exists in DataFrame."""
    from ..exceptions import ColumnNotFoundError

    if column not in df.columns:
        raise ColumnNotFoundError(column, df.columns.tolist())


# Implementation: Type conversion with error handling for int, float, string, datetime, boolean
def safe_type_conversion(series: pd.Series, target_type: str) -> pd.Series:
    """Convert pandas Series to target type with error handling."""
    try:
        if target_type == "int":
            return pd.to_numeric(series, errors="coerce").astype("Int64")
        elif target_type == "float":
            return pd.to_numeric(series, errors="coerce")
        elif target_type == "string":
            return series.astype(str)
        elif target_type == "datetime":
            return pd.to_datetime(series, errors="coerce")
        elif target_type == "boolean":
            return series.astype(bool)
        else:
            raise ValueError(f"Unsupported type: {target_type}")
    except (ValueError, TypeError, OverflowError) as e:
        raise ValueError(f"Failed to convert to {target_type}: {e}") from e
