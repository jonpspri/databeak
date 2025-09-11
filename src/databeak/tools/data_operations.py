"""Core data operations and utilities for CSV Editor."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..exceptions import InvalidRowIndexError, NoDataLoadedError, SessionNotFoundError
from ..models import get_session_manager


def create_data_preview_with_indices(df: pd.DataFrame, num_rows: int = 5) -> dict[str, Any]:
    """Create a data preview with row indices for AI accessibility."""
    preview_df = df.head(num_rows)

    # Create records with row indices
    preview_records = []
    for _, (row_idx, row) in enumerate(preview_df.iterrows()):
        # Handle pandas index types safely
        row_index_val = row_idx if isinstance(row_idx, int) else 0
        record = {"__row_index__": row_index_val}  # Include original row index
        record.update(row.to_dict())

        # Handle pandas/numpy types for JSON serialization
        for key, value in record.items():
            if key == "__row_index__":
                continue
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, pd.Timestamp):
                record[key] = str(value)
            elif hasattr(value, "item"):
                record[key] = value.item()

        preview_records.append(record)

    return {
        "records": preview_records,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "preview_rows": len(preview_records),
    }


def get_data_summary(session_id: str) -> dict[str, Any]:
    """Get a summary of the data in a session."""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if not session:
        raise SessionNotFoundError(session_id)
    if session.df is None:
        raise NoDataLoadedError(session_id)

    df = session.df

    return {
        "session_id": session_id,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "null_counts": df.isnull().sum().to_dict(),
        "preview": create_data_preview_with_indices(df, 10),
    }


def validate_row_index(df: pd.DataFrame, row_index: int) -> None:
    """Validate that a row index is within bounds."""
    if row_index < 0 or row_index >= len(df):
        raise InvalidRowIndexError(row_index, len(df) - 1)


def validate_column_exists(df: pd.DataFrame, column: str) -> None:
    """Validate that a column exists in the dataframe."""
    from ..exceptions import ColumnNotFoundError

    if column not in df.columns:
        raise ColumnNotFoundError(column, df.columns.tolist())


def safe_type_conversion(series: pd.Series, target_type: str) -> pd.Series:
    """Safely convert a pandas Series to a target type."""
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
    except Exception as e:
        raise ValueError(f"Failed to convert to {target_type}: {e}") from e
