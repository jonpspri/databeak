"""Unit tests for data_operations.py.

Tests for create_data_preview_with_indices function which is used by servers
to generate preview data for display to users.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from databeak.services.data_operations import create_data_preview_with_indices


class TestCreateDataPreviewWithIndices:
    """Test create_data_preview_with_indices function with various data types."""

    def test_basic_dataframe(self) -> None:
        """Test with simple dataframe."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [30, 25, 35],
                "salary": [60000.0, 50000.0, 70000.0],
            },
        )

        result = create_data_preview_with_indices(df, 2)

        assert result["total_rows"] == 3
        assert result["total_columns"] == 3
        assert result["preview_rows"] == 2
        assert result["columns"] == ["name", "age", "salary"]
        assert len(result["records"]) == 2

        # Check first record structure
        record = result["records"][0]
        assert record["__row_index__"] == 0
        assert record["name"] == "Alice"
        assert record["age"] == 30
        assert record["salary"] == 60000.0

    def test_with_nan_values(self) -> None:
        """Test handling of NaN values."""
        df = pd.DataFrame({"col1": [1, np.nan, 3], "col2": ["a", "b", np.nan]})

        result = create_data_preview_with_indices(df, 3)

        records = result["records"]
        assert records[1]["col1"] is None  # NaN converted to None
        assert records[2]["col2"] is None  # NaN converted to None

    def test_with_timestamp_values(self) -> None:
        """Test handling of pandas Timestamp objects."""
        timestamps = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
        df = pd.DataFrame({"date": timestamps, "value": [1, 2, 3]})

        result = create_data_preview_with_indices(df, 2)

        records = result["records"]
        # Timestamps should be converted to strings
        assert isinstance(records[0]["date"], str)
        assert "2023-01-01" in records[0]["date"]

    def test_with_numpy_types(self) -> None:
        """Test handling of numpy scalar types."""
        df = pd.DataFrame(
            {
                "int_col": np.array([1, 2, 3], dtype=np.int64),
                "float_col": np.array([1.1, 2.2, 3.3], dtype=np.float64),
                "bool_col": np.array([True, False, True], dtype=bool),
            },
        )

        result = create_data_preview_with_indices(df, 2)

        records = result["records"]
        # Check that numpy types are converted using .item()
        assert isinstance(records[0]["int_col"], int)
        assert isinstance(records[0]["float_col"], float)
        assert isinstance(records[0]["bool_col"], bool)

    def test_with_non_integer_index(self) -> None:
        """Test handling of non-integer row indices."""
        df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"]},
            index=["row1", "row2", "row3"],
        )

        result = create_data_preview_with_indices(df, 2)

        records = result["records"]
        # Non-integer indices should default to 0
        assert records[0]["__row_index__"] == 0
        assert records[1]["__row_index__"] == 0

    def test_with_complex_column_names(self) -> None:
        """Test handling of complex column names."""
        df = pd.DataFrame(
            {
                123: [1, 2, 3],  # Numeric column name
                "spaced column": ["a", "b", "c"],  # Spaced column name
                ("tuple", "col"): [True, False, True],  # Tuple column name
            },
        )

        result = create_data_preview_with_indices(df, 2)

        records = result["records"]
        # All column names should be converted to strings
        assert "123" in records[0]
        assert "spaced column" in records[0]
        assert "('tuple', 'col')" in records[0]

    def test_empty_dataframe(self) -> None:
        """Test with empty dataframe."""
        df = pd.DataFrame()

        result = create_data_preview_with_indices(df, 5)

        assert result["total_rows"] == 0
        assert result["total_columns"] == 0
        assert result["preview_rows"] == 0
        assert result["records"] == []
        assert result["columns"] == []

    def test_more_rows_requested_than_available(self) -> None:
        """Test when requesting more preview rows than available."""
        df = pd.DataFrame({"col1": [1, 2]})

        result = create_data_preview_with_indices(df, 10)

        assert result["total_rows"] == 2
        assert result["preview_rows"] == 2
        assert len(result["records"]) == 2

    def test_zero_preview_rows(self) -> None:
        """Test with zero preview rows requested."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        result = create_data_preview_with_indices(df, 0)

        assert result["total_rows"] == 3
        assert result["preview_rows"] == 0
        assert len(result["records"]) == 0

    @pytest.mark.parametrize(
        ("special_value", "expected"),
        [
            (np.inf, float("inf")),
            (-np.inf, float("-inf")),
        ],
    )
    def test_special_values(self, special_value: float, expected: float) -> None:
        """Test handling of special pandas/numpy values."""
        df = pd.DataFrame({"col": [1, special_value, 3]})

        result = create_data_preview_with_indices(df, 3)

        record_value = result["records"][1]["col"]
        assert record_value == expected

    def test_pd_na_values(self) -> None:
        """Test handling of pandas NA values separately."""
        df = pd.DataFrame({"col": [1, pd.NA, 3]})

        result = create_data_preview_with_indices(df, 3)

        record_value = result["records"][1]["col"]
        assert record_value is None
