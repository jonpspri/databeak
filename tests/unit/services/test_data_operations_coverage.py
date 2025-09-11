"""Comprehensive coverage tests for data_operations module."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.databeak.exceptions import (
    ColumnNotFoundError,
    InvalidRowIndexError,
    NoDataLoadedError,
    SessionNotFoundError,
)
from src.databeak.services.data_operations import (
    create_data_preview_with_indices,
    get_data_summary,
    safe_type_conversion,
    validate_column_exists,
    validate_row_index,
)


class TestCreateDataPreview:
    """Test data preview creation functionality."""

    def test_create_data_preview_basic(self):
        """Test basic data preview creation."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "salary": [50000, 60000, 70000],
            }
        )

        result = create_data_preview_with_indices(df)

        assert isinstance(result, dict)
        assert "records" in result
        assert "total_rows" in result
        assert "total_columns" in result
        assert "columns" in result
        assert "preview_rows" in result

        assert result["total_rows"] == 3
        assert result["total_columns"] == 3
        assert result["preview_rows"] == 3
        assert len(result["records"]) == 3

    def test_create_data_preview_with_custom_rows(self):
        """Test data preview with custom number of rows."""
        df = pd.DataFrame({"col1": range(10), "col2": [f"value_{i}" for i in range(10)]})

        result = create_data_preview_with_indices(df, num_rows=3)

        assert result["preview_rows"] == 3
        assert len(result["records"]) == 3
        assert result["total_rows"] == 10

    def test_create_data_preview_with_row_indices(self):
        """Test data preview includes row indices."""
        df = pd.DataFrame({"data": ["a", "b", "c"]})

        result = create_data_preview_with_indices(df)

        for i, record in enumerate(result["records"]):
            assert "__row_index__" in record
            assert record["__row_index__"] == i

    def test_create_data_preview_with_null_values(self):
        """Test data preview handles null values."""
        df = pd.DataFrame(
            {"col1": [1, None, 3], "col2": ["a", "b", None], "col3": [1.1, 2.2, float("nan")]}
        )

        result = create_data_preview_with_indices(df)

        # Check null handling
        records = result["records"]
        assert records[1]["col1"] is None  # None -> None
        assert records[2]["col2"] is None  # None -> None

    def test_create_data_preview_with_pandas_types(self):
        """Test data preview handles pandas/numpy types."""
        df = pd.DataFrame(
            {
                "int_col": pd.array([1, 2, 3], dtype="Int64"),
                "float_col": pd.array([1.1, 2.2, 3.3], dtype="float64"),
                "bool_col": pd.array([True, False, True], dtype="boolean"),
            }
        )

        result = create_data_preview_with_indices(df)

        # Should convert pandas types to Python types for JSON serialization
        for record in result["records"]:
            assert isinstance(record["int_col"], int | type(None))
            assert isinstance(record["float_col"], float | type(None))
            assert isinstance(record["bool_col"], bool | type(None))

    def test_create_data_preview_non_integer_index(self):
        """Test data preview with non-integer pandas index."""
        df = pd.DataFrame({"data": ["a", "b", "c"]})
        df.index = ["x", "y", "z"]  # String index

        result = create_data_preview_with_indices(df)

        # Should handle non-integer index gracefully
        for record in result["records"]:
            assert "__row_index__" in record
            assert isinstance(record["__row_index__"], int)

    def test_create_data_preview_empty_dataframe(self):
        """Test data preview with empty dataframe."""
        df = pd.DataFrame(columns=["col1", "col2"])

        result = create_data_preview_with_indices(df)

        assert result["total_rows"] == 0
        assert result["total_columns"] == 2
        assert result["preview_rows"] == 0
        assert len(result["records"]) == 0


class TestGetDataSummary:
    """Test data summary functionality."""

    @patch("src.databeak.tools.data_operations.get_session_manager")
    def test_get_data_summary_basic(self, mock_get_session_manager):
        """Test basic data summary retrieval."""
        # Mock session with data using property-based API
        mock_session = Mock()
        mock_session.df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

        mock_session_manager = Mock()
        mock_session_manager.get_session.return_value = mock_session
        mock_get_session_manager.return_value = mock_session_manager

        result = get_data_summary("test_session")

        assert isinstance(result, dict)
        assert result["session_id"] == "test_session"
        assert "shape" in result
        assert "columns" in result
        assert "dtypes" in result
        assert "memory_usage_mb" in result
        assert "null_counts" in result
        assert "preview" in result

    @patch("src.databeak.tools.data_operations.get_session_manager")
    def test_get_data_summary_session_not_found(self, mock_get_session_manager):
        """Test data summary when session not found."""
        mock_session_manager = Mock()
        mock_session_manager.get_session.return_value = None
        mock_get_session_manager.return_value = mock_session_manager

        with pytest.raises(SessionNotFoundError):
            get_data_summary("nonexistent_session")

    @patch("src.databeak.tools.data_operations.get_session_manager")
    def test_get_data_summary_no_data_loaded(self, mock_get_session_manager):
        """Test data summary when no data loaded."""
        mock_session = Mock()
        mock_session.df = None

        mock_session_manager = Mock()
        mock_session_manager.get_session.return_value = mock_session
        mock_get_session_manager.return_value = mock_session_manager

        with pytest.raises(NoDataLoadedError):
            get_data_summary("session_with_no_data")

    @patch("src.databeak.tools.data_operations.get_session_manager")
    def test_get_data_summary_with_nulls_and_dtypes(self, mock_get_session_manager):
        """Test data summary with various data types and nulls."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, None],
                "float_col": [1.1, None, 3.3],
                "str_col": ["a", "b", None],
                "bool_col": [True, False, None],
            }
        )

        mock_session = Mock()
        mock_session.df = df

        mock_session_manager = Mock()
        mock_session_manager.get_session.return_value = mock_session
        mock_get_session_manager.return_value = mock_session_manager

        result = get_data_summary("test_session")

        # Check dtypes are converted to strings
        assert isinstance(result["dtypes"], dict)
        for _col, dtype in result["dtypes"].items():
            assert isinstance(dtype, str)

        # Check null counts
        assert isinstance(result["null_counts"], dict)
        assert result["null_counts"]["int_col"] == 1
        assert result["null_counts"]["float_col"] == 1
        assert result["null_counts"]["str_col"] == 1

    @patch("src.databeak.tools.data_operations.get_session_manager")
    def test_get_data_summary_memory_calculation(self, mock_get_session_manager):
        """Test memory usage calculation in data summary."""
        # Create larger dataframe for memory testing
        df = pd.DataFrame({"data": range(1000), "text": [f"text_{i}" for i in range(1000)]})

        mock_session = Mock()
        mock_session.df = df

        mock_session_manager = Mock()
        mock_session_manager.get_session.return_value = mock_session
        mock_get_session_manager.return_value = mock_session_manager

        result = get_data_summary("test_session")

        assert "memory_usage_mb" in result
        assert isinstance(result["memory_usage_mb"], int | float)
        assert result["memory_usage_mb"] > 0


class TestValidationFunctions:
    """Test validation utility functions."""

    def test_validate_row_index_valid(self):
        """Test valid row index validation."""
        df = pd.DataFrame({"col1": range(5)})

        # Valid indices should not raise
        validate_row_index(df, 0)
        validate_row_index(df, 4)  # last valid index
        validate_row_index(df, 2)  # middle index

    def test_validate_row_index_negative(self):
        """Test negative row index validation."""
        df = pd.DataFrame({"col1": range(5)})

        with pytest.raises(InvalidRowIndexError):
            validate_row_index(df, -1)

    def test_validate_row_index_too_large(self):
        """Test row index too large validation."""
        df = pd.DataFrame({"col1": range(5)})

        with pytest.raises(InvalidRowIndexError):
            validate_row_index(df, 5)  # index == len(df)

        with pytest.raises(InvalidRowIndexError):
            validate_row_index(df, 10)  # way too large

    def test_validate_row_index_empty_dataframe(self):
        """Test row index validation with empty dataframe."""
        df = pd.DataFrame(columns=["col1"])

        with pytest.raises(InvalidRowIndexError):
            validate_row_index(df, 0)

    def test_validate_column_exists_valid(self):
        """Test valid column existence validation."""
        df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col_with_underscore": [1.1, 2.2, 3.3]}
        )

        # Valid columns should not raise
        validate_column_exists(df, "col1")
        validate_column_exists(df, "col2")
        validate_column_exists(df, "col_with_underscore")

    def test_validate_column_exists_invalid(self):
        """Test invalid column existence validation."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        with pytest.raises(ColumnNotFoundError):
            validate_column_exists(df, "nonexistent_column")

    def test_validate_column_exists_case_sensitive(self):
        """Test column validation is case sensitive."""
        df = pd.DataFrame({"Column1": [1, 2, 3]})

        validate_column_exists(df, "Column1")  # Should pass

        with pytest.raises(ColumnNotFoundError):
            validate_column_exists(df, "column1")  # Different case


class TestSafeTypeConversion:
    """Test safe type conversion functionality."""

    def test_safe_type_conversion_to_int(self):
        """Test safe conversion to integer type."""
        series = pd.Series([1.0, 2.0, 3.0])

        result = safe_type_conversion(series, "int")

        assert result.dtype == "Int64"  # Nullable integer
        assert result.tolist() == [1, 2, 3]

    def test_safe_type_conversion_to_int_with_nulls(self):
        """Test safe conversion to int with null values."""
        series = pd.Series([1.0, None, 3.0])

        result = safe_type_conversion(series, "int")

        assert result.dtype == "Int64"  # Nullable integer
        assert pd.isna(result.iloc[1])

    def test_safe_type_conversion_to_float(self):
        """Test safe conversion to float type."""
        series = pd.Series(["1.1", "2.2", "3.3"])

        result = safe_type_conversion(series, "float")

        assert result.dtype == "float64"
        assert result.tolist() == [1.1, 2.2, 3.3]

    def test_safe_type_conversion_to_string(self):
        """Test safe conversion to string type."""
        series = pd.Series([1, 2, 3])

        result = safe_type_conversion(series, "string")

        assert result.dtype == "object"
        assert result.tolist() == ["1", "2", "3"]

    def test_safe_type_conversion_to_datetime(self):
        """Test safe conversion to datetime type."""
        series = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])

        result = safe_type_conversion(series, "datetime")

        assert pd.api.types.is_datetime64_any_dtype(result)

    def test_safe_type_conversion_to_boolean(self):
        """Test safe conversion to boolean type."""
        series = pd.Series([1, 0, 1])

        result = safe_type_conversion(series, "boolean")

        assert result.dtype == "bool"
        assert result.tolist() == [True, False, True]

    def test_safe_type_conversion_invalid_int(self):
        """Test safe conversion with invalid integers."""
        series = pd.Series(["not", "a", "number"])

        result = safe_type_conversion(series, "int")

        # Should coerce errors to NaN/None
        assert result.dtype == "Int64"
        assert pd.isna(result).all()

    def test_safe_type_conversion_invalid_float(self):
        """Test safe conversion with invalid floats."""
        series = pd.Series(["not", "a", "float"])

        result = safe_type_conversion(series, "float")

        # Should coerce errors to NaN
        assert result.dtype == "float64"
        assert pd.isna(result).all()

    def test_safe_type_conversion_invalid_datetime(self):
        """Test safe conversion with invalid datetime."""
        series = pd.Series(["not-a-date", "invalid", "datetime"])

        result = safe_type_conversion(series, "datetime")

        # Should coerce errors to NaT
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert pd.isna(result).all()

    def test_safe_type_conversion_unsupported_type(self):
        """Test safe conversion with unsupported type."""
        series = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="Unsupported type"):
            safe_type_conversion(series, "unsupported_type")

    def test_safe_type_conversion_exception_handling(self):
        """Test safe conversion exception handling."""
        # Create a series that might cause conversion issues
        series = pd.Series([1, 2, 3])

        with patch("pandas.to_numeric", side_effect=Exception("Conversion error")):
            with pytest.raises(ValueError, match="Failed to convert"):
                safe_type_conversion(series, "int")


class TestDataOperationsEdgeCases:
    """Test edge cases and error conditions."""

    def test_create_data_preview_very_large_dataframe(self):
        """Test data preview with large dataframe."""
        # Create large dataframe but only preview few rows
        df = pd.DataFrame({"data": range(100000)})

        result = create_data_preview_with_indices(df, num_rows=2)

        assert result["total_rows"] == 100000
        assert result["preview_rows"] == 2
        assert len(result["records"]) == 2

    def test_create_data_preview_zero_rows(self):
        """Test data preview with zero rows requested."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        result = create_data_preview_with_indices(df, num_rows=0)

        assert result["total_rows"] == 3
        assert result["preview_rows"] == 0
        assert len(result["records"]) == 0

    def test_safe_type_conversion_empty_series(self):
        """Test type conversion with empty series."""
        series = pd.Series([], dtype=object)

        result = safe_type_conversion(series, "int")

        assert len(result) == 0
        assert result.dtype == "Int64"

    def test_validation_with_special_column_names(self):
        """Test validation with special column names."""
        df = pd.DataFrame(
            {
                "col with spaces": [1, 2, 3],
                "col-with-hyphens": [1, 2, 3],
                "123numeric_start": [1, 2, 3],
                "": [1, 2, 3],  # empty column name
            }
        )

        # Should handle special column names
        validate_column_exists(df, "col with spaces")
        validate_column_exists(df, "col-with-hyphens")
        validate_column_exists(df, "123numeric_start")
        validate_column_exists(df, "")  # empty name
