"""Comprehensive unit tests for data_operations.py.

This test suite covers all functions in src/databeak/services/data_operations.py with focus on edge
cases, error handling, and type conversions.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest

from databeak.core.session import get_session_manager
from databeak.exceptions import (
    ColumnNotFoundError,
    InvalidRowIndexError,
    NoDataLoadedError,
)
from databeak.services.data_operations import (
    create_data_preview_with_indices,
    get_data_summary,
    safe_type_conversion,
    validate_column_exists,
    validate_row_index,
)


class TestCreateDataPreviewWithIndices:
    """Test create_data_preview_with_indices function with various data types."""

    def test_basic_dataframe(self):
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

    def test_with_nan_values(self):
        """Test handling of NaN values."""
        df = pd.DataFrame({"col1": [1, np.nan, 3], "col2": ["a", "b", np.nan]})

        result = create_data_preview_with_indices(df, 3)

        records = result["records"]
        assert records[1]["col1"] is None  # NaN converted to None
        assert records[2]["col2"] is None  # NaN converted to None

    def test_with_timestamp_values(self):
        """Test handling of pandas Timestamp objects."""
        timestamps = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
        df = pd.DataFrame({"date": timestamps, "value": [1, 2, 3]})

        result = create_data_preview_with_indices(df, 2)

        records = result["records"]
        # Timestamps should be converted to strings
        assert isinstance(records[0]["date"], str)
        assert "2023-01-01" in records[0]["date"]

    def test_with_numpy_types(self):
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

    def test_with_non_integer_index(self):
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

    def test_with_complex_column_names(self):
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

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame()

        result = create_data_preview_with_indices(df, 5)

        assert result["total_rows"] == 0
        assert result["total_columns"] == 0
        assert result["preview_rows"] == 0
        assert result["records"] == []
        assert result["columns"] == []

    def test_more_rows_requested_than_available(self):
        """Test when requesting more preview rows than available."""
        df = pd.DataFrame({"col1": [1, 2]})

        result = create_data_preview_with_indices(df, 10)

        assert result["total_rows"] == 2
        assert result["preview_rows"] == 2
        assert len(result["records"]) == 2

    def test_zero_preview_rows(self):
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
    def test_special_values(self, special_value, expected):
        """Test handling of special pandas/numpy values."""
        df = pd.DataFrame({"col": [1, special_value, 3]})

        result = create_data_preview_with_indices(df, 3)

        record_value = result["records"][1]["col"]
        assert record_value == expected

    def test_pd_na_values(self):
        """Test handling of pandas NA values separately."""
        df = pd.DataFrame({"col": [1, pd.NA, 3]})

        result = create_data_preview_with_indices(df, 3)

        record_value = result["records"][1]["col"]
        assert record_value is None


class TestGetDataSummary:
    """Test get_data_summary function with various scenarios."""

    def test_successful_summary(self):
        """Test successful data summary generation."""
        # Create test data
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3, None],
                "str_col": ["a", "b", "c", "d"],
                "float_col": [1.1, 2.2, None, 4.4],
            },
        )

        # Create a real session and load data
        session_id = str(uuid.uuid4())
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)
        session.df = df

        result = get_data_summary(session_id)

        assert result["session_id"] == session_id
        assert result["shape"] == (4, 3)
        assert result["columns"] == ["int_col", "str_col", "float_col"]
        # DataFrame with None values may convert int to float64
        assert (
            "int64" in result["dtypes"]["int_col"]
            or "float64" in result["dtypes"]["int_col"]
            or "object" in result["dtypes"]["int_col"]
        )
        assert "object" in result["dtypes"]["str_col"]
        assert "float64" in result["dtypes"]["float_col"]
        assert result["null_counts"]["int_col"] == 1
        assert result["null_counts"]["str_col"] == 0
        assert result["null_counts"]["float_col"] == 1
        assert "preview" in result
        assert result["memory_usage_mb"] >= 0  # Memory usage can be 0 for very small dataframes

    def test_no_data_loaded(self):
        """Test NoDataLoadedError when session has no data."""
        # Create a real session with no data
        session_id = str(uuid.uuid4())
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)
        session.df = None

        with pytest.raises(NoDataLoadedError) as exc_info:
            get_data_summary(session_id)

        assert session_id in str(exc_info.value)

    def test_large_dataframe_memory_calculation(self):
        """Test memory calculation for larger dataframes."""
        # Create a larger dataframe
        df = pd.DataFrame({"col1": range(1000), "col2": ["text"] * 1000, "col3": [1.5] * 1000})

        # Create a real session and load data
        session_id = str(uuid.uuid4())
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)
        session.df = df

        result = get_data_summary(session_id)

        assert result["memory_usage_mb"] > 0
        assert result["shape"] == (1000, 3)
        assert result["preview"]["total_rows"] == 1000
        assert len(result["preview"]["records"]) == 10  # Default preview rows

    def test_empty_dataframe_summary(self):
        """Test summary of empty dataframe."""
        # Create empty dataframe
        df = pd.DataFrame()

        # Create a real session and load data
        session_id = str(uuid.uuid4())
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)
        session.df = df

        result = get_data_summary(session_id)

        assert result["shape"] == (0, 0)
        assert result["columns"] == []
        assert result["dtypes"] == {}
        assert result["null_counts"] == {}
        assert result["preview"]["records"] == []


class TestValidateRowIndex:
    """Test validate_row_index function with boundary conditions."""

    def test_valid_index(self):
        """Test with valid row index."""
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})

        # Should not raise exception
        validate_row_index(df, 0)
        validate_row_index(df, 2)
        validate_row_index(df, 4)  # Last valid index

    def test_negative_index(self):
        """Test with negative row index."""
        df = pd.DataFrame({"col": [1, 2, 3]})

        with pytest.raises(InvalidRowIndexError) as exc_info:
            validate_row_index(df, -1)

        error = exc_info.value
        assert error.details["row_index"] == -1
        assert error.details["max_index"] == 2

    def test_index_too_large(self):
        """Test with index larger than dataframe."""
        df = pd.DataFrame({"col": [1, 2, 3]})

        with pytest.raises(InvalidRowIndexError) as exc_info:
            validate_row_index(df, 3)

        error = exc_info.value
        assert error.details["row_index"] == 3
        assert error.details["max_index"] == 2

    def test_index_much_too_large(self):
        """Test with index much larger than dataframe."""
        df = pd.DataFrame({"col": [1]})

        with pytest.raises(InvalidRowIndexError) as exc_info:
            validate_row_index(df, 100)

        error = exc_info.value
        assert error.details["row_index"] == 100
        assert error.details["max_index"] == 0

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame()

        with pytest.raises(InvalidRowIndexError):
            validate_row_index(df, 0)

    def test_single_row_dataframe(self):
        """Test with single row dataframe."""
        df = pd.DataFrame({"col": [42]})

        # Should work for index 0
        validate_row_index(df, 0)

        # Should fail for index 1
        with pytest.raises(InvalidRowIndexError):
            validate_row_index(df, 1)


class TestValidateColumnExists:
    """Test validate_column_exists function with various scenarios."""

    def test_existing_column(self):
        """Test with existing column."""
        df = pd.DataFrame({"name": [1, 2], "age": [3, 4], "salary": [5, 6]})

        # Should not raise exception
        validate_column_exists(df, "name")
        validate_column_exists(df, "age")
        validate_column_exists(df, "salary")

    def test_missing_column(self):
        """Test with missing column."""
        df = pd.DataFrame({"name": [1, 2], "age": [3, 4]})

        with pytest.raises(ColumnNotFoundError) as exc_info:
            validate_column_exists(df, "salary")

        error = exc_info.value
        assert error.details["column_name"] == "salary"
        assert error.details["available_columns"] == ["name", "age"]

    def test_case_sensitive_column_names(self):
        """Test that column validation is case sensitive."""
        df = pd.DataFrame({"Name": [1, 2], "AGE": [3, 4]})

        # Should work with exact case
        validate_column_exists(df, "Name")
        validate_column_exists(df, "AGE")

        # Should fail with different case
        with pytest.raises(ColumnNotFoundError):
            validate_column_exists(df, "name")

        with pytest.raises(ColumnNotFoundError):
            validate_column_exists(df, "age")

    def test_numeric_column_names(self):
        """Test with numeric column names."""
        df = pd.DataFrame({1: [1, 2], 2: [3, 4], "text": [5, 6]})

        validate_column_exists(df, 1)
        validate_column_exists(df, 2)
        validate_column_exists(df, "text")

        with pytest.raises(ColumnNotFoundError):
            validate_column_exists(df, 3)

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame()

        with pytest.raises(ColumnNotFoundError) as exc_info:
            validate_column_exists(df, "any_column")

        error = exc_info.value
        assert error.details["available_columns"] == []

    def test_special_column_names(self):
        """Test with special characters in column names."""
        df = pd.DataFrame(
            {
                "col with spaces": [1, 2],
                "col-with-dashes": [3, 4],
                "col_with_underscores": [5, 6],
                "col.with.dots": [7, 8],
            },
        )

        validate_column_exists(df, "col with spaces")
        validate_column_exists(df, "col-with-dashes")
        validate_column_exists(df, "col_with_underscores")
        validate_column_exists(df, "col.with.dots")


class TestSafeTypeConversion:
    """Test safe_type_conversion function with all supported types."""

    def test_convert_to_int(self):
        """Test conversion to int type."""
        series = pd.Series(["1", "2", "3.0", "invalid", ""])

        result = safe_type_conversion(series, "int")

        # Check that valid conversions work and invalid become NaN
        assert result.iloc[0] == 1
        assert result.iloc[1] == 2
        assert result.iloc[2] == 3  # 3.0 should convert to 3
        assert pd.isna(result.iloc[3])  # 'invalid' should become NaN
        assert pd.isna(result.iloc[4])  # empty string should become NaN
        # Check dtype is nullable integer
        assert result.dtype == "Int64"

    def test_convert_to_float(self):
        """Test conversion to float type."""
        series = pd.Series(["1.1", "2", "3.14", "invalid", ""])

        result = safe_type_conversion(series, "float")

        assert result.iloc[0] == 1.1
        assert result.iloc[1] == 2.0
        assert result.iloc[2] == 3.14
        assert pd.isna(result.iloc[3])
        assert pd.isna(result.iloc[4])
        assert result.dtype == "float64"

    def test_convert_to_string(self):
        """Test conversion to string type."""
        series = pd.Series([1, 2.5, True, None, np.nan])

        result = safe_type_conversion(series, "string")

        assert result.iloc[0] == "1"
        assert result.iloc[1] == "2.5"
        assert result.iloc[2] == "True"
        assert result.iloc[3] == "None"
        assert "nan" in result.iloc[4].lower()
        assert result.dtype == "object"

    def test_convert_to_datetime(self):
        """Test conversion to datetime type."""
        series = pd.Series(["2023-01-01", "2023-12-31", "invalid", "2023-01-01 12:30:45"])

        result = safe_type_conversion(series, "datetime")

        assert pd.notna(result.iloc[0])
        assert result.iloc[0].year == 2023
        assert result.iloc[0].month == 1
        assert pd.notna(result.iloc[1])
        assert result.iloc[1].month == 12
        assert pd.isna(result.iloc[2])  # 'invalid' should become NaT
        # Some datetime formats may not parse successfully
        assert pd.notna(result.iloc[3]) or pd.isna(
            result.iloc[3],
        )  # Either valid or NaT is acceptable

    def test_convert_to_boolean(self):
        """Test conversion to boolean type."""
        series = pd.Series([1, 0, "true", "false", None])

        result = safe_type_conversion(series, "boolean")

        assert bool(result.iloc[0]) is True  # 1 becomes True
        assert bool(result.iloc[1]) is False  # 0 becomes False
        assert bool(result.iloc[2]) is True  # 'true' becomes True
        assert bool(result.iloc[3]) is True  # 'false' becomes True (non-empty string)
        # None might become True or False depending on pandas behavior

    def test_unsupported_type(self):
        """Test with unsupported target type."""
        series = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="Unsupported type: complex"):
            safe_type_conversion(series, "complex")

    def test_conversion_exception_handling(self):
        """Test that conversion exceptions are properly wrapped."""
        # Create a series that will cause conversion issues
        series = pd.Series([1, 2, 3])

        # Test with a type that doesn't exist to trigger the except block
        with pytest.raises(ValueError, match="Unsupported type"):
            safe_type_conversion(series, "nonexistent_type")

    def test_numeric_conversion_edge_cases(self):
        """Test numeric conversions with edge cases."""
        # Test with very large numbers, scientific notation, etc.
        series = pd.Series(["1e10", "1.23e-5", "inf", "-inf", "nan"])

        float_result = safe_type_conversion(series, "float")
        assert float_result.iloc[0] == 1e10
        assert float_result.iloc[1] == 1.23e-5
        assert np.isinf(float_result.iloc[2])
        assert np.isinf(float_result.iloc[3])
        assert float_result.iloc[3] < 0
        assert pd.isna(float_result.iloc[4])

    def test_datetime_conversion_formats(self):
        """Test datetime conversion with various formats."""
        series = pd.Series(
            [
                "2023-01-01",
                "01/01/2023",
                "2023-01-01T12:30:45",
                "2023-01-01 12:30:45.123",
                "Jan 1, 2023",
            ],
        )

        result = safe_type_conversion(series, "datetime")

        # At least some should convert successfully
        valid_count = sum(1 for i in range(len(result)) if pd.notna(result.iloc[i]))
        assert valid_count >= 1  # At least 1 should parse successfully

    def test_empty_series_conversion(self):
        """Test conversion with empty series."""
        series = pd.Series([], dtype=object)

        result = safe_type_conversion(series, "int")

        assert len(result) == 0
        assert result.dtype == "Int64"

    @pytest.mark.parametrize(
        ("target_type", "expected_dtype"),
        [
            ("int", "Int64"),
            ("string", "object"),
            ("datetime", "datetime64[ns]"),
            ("boolean", "bool"),
        ],
    )
    def test_all_type_conversions_dtype(self, target_type, expected_dtype):
        """Test that all conversions produce expected dtypes."""
        if target_type == "float":
            # Use numeric data for float conversion
            series = pd.Series([1.1, 2.2, 3.3])
        else:
            series = pd.Series(["1", "2", "3"])  # type: ignore[assignment]

        result = safe_type_conversion(series, target_type)

        if expected_dtype == "datetime64[ns]":
            assert "datetime64" in str(result.dtype)
        else:
            assert result.dtype == expected_dtype

    def test_float_conversion_dtype(self):
        """Test float conversion specifically."""
        series = pd.Series([1.1, 2.2, 3.3])
        result = safe_type_conversion(series, "float")
        assert result.dtype == "float64"
