"""Core data operations and utilities for CSV data manipulation."""

from __future__ import annotations

import pandas as pd

from databeak.models.typed_dicts import CellValue, DataPreviewResult


def create_data_preview_with_indices(df: pd.DataFrame, num_rows: int = 5) -> DataPreviewResult:
    """Create data preview with row indices and metadata.

    This function is used by servers to generate preview data for display to users.
    It converts DataFrame rows to dictionaries with proper type handling for
    pandas/numpy types and includes row index information.

    Args:
        df: DataFrame to preview
        num_rows: Number of rows to include in preview (default: 5)

    Returns:
        DataPreviewResult with preview records and metadata

    """
    preview_df = df.head(num_rows)

    # Create records with row indices
    preview_records = []
    for _, (row_idx, row) in enumerate(preview_df.iterrows()):
        # Handle pandas index types safely
        row_index_val = row_idx if isinstance(row_idx, int) else 0
        # Convert all keys to strings and handle pandas/numpy types
        record: dict[str, CellValue] = {
            "__row_index__": row_index_val,
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
