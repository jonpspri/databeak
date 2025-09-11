"""Comprehensive unit tests for statistics_server module to improve coverage."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

from src.databeak.servers.statistics_server import (
    ColumnStatisticsResult,
    CorrelationResult,
    StatisticsResult,
    ValueCountsResult,
    get_column_statistics,
    get_correlation_matrix,
    get_statistics,
    get_value_counts,
)


@pytest.fixture
def mock_session_with_data():
    """Create mock session with diverse test data."""
    session = Mock()
    df = pd.DataFrame(
        {
            "numeric1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "numeric2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "numeric3": [1.1, 2.2, 3.3, 4.4, 5.5, np.nan, 7.7, 8.8, 9.9, 10.0],
            "categorical": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
            "mixed": [1, "two", 3, "four", 5, "six", 7, "eight", 9, "ten"],
            "dates": pd.date_range("2024-01-01", periods=10),
            "boolean": [True, False, True, False, True, False, True, False, True, False],
            "all_null": [None] * 10,
            "mostly_null": [1, None, None, None, 5, None, None, None, None, 10],
        }
    )
    # Configure mock to use new API
    session._df = df
    type(session).df = property(
        fget=lambda self: self._df, fset=lambda self, value: setattr(self, "_df", value)
    )
    session.has_data.return_value = True
    session.record_operation = Mock()
    return session


@pytest.fixture
def mock_empty_session():
    """Create mock session with empty dataframe."""
    session = Mock()
    df = pd.DataFrame()
    # Configure mock to use new API
    session._df = df
    type(session).df = property(
        fget=lambda self: self._df, fset=lambda self, value: setattr(self, "_df", value)
    )
    session.has_data.return_value = True
    session.record_operation = Mock()
    return session


@pytest.fixture
def mock_manager(mock_session_with_data):
    """Create mock session manager."""
    with patch("src.databeak.servers.statistics_server.get_session_manager") as manager:
        manager.return_value.get_session.return_value = mock_session_with_data
        yield manager


@pytest.mark.asyncio
class TestGetStatistics:
    """Test get_statistics function comprehensively."""

    async def test_statistics_all_columns(self, mock_manager):
        """Test getting statistics for all columns."""
        result = await get_statistics("test-session")

        assert isinstance(result, StatisticsResult)
        assert result.success is True
        assert len(result.statistics) > 0
        assert result.total_rows == 10
        assert result.total_columns == 9

        # Check numeric column stats
        numeric1_stats = next(s for s in result.statistics if s["column"] == "numeric1")
        assert numeric1_stats["mean"] == 5.5
        assert numeric1_stats["median"] == 5.5
        assert numeric1_stats["std"] > 0
        assert numeric1_stats["min"] == 1
        assert numeric1_stats["max"] == 10

    async def test_statistics_specific_columns(self, mock_manager):
        """Test getting statistics for specific columns."""
        result = await get_statistics("test-session", columns=["numeric1", "numeric2"])

        assert result.success is True
        assert len(result.statistics) == 2
        assert all(s["column"] in ["numeric1", "numeric2"] for s in result.statistics)

    async def test_statistics_with_nulls(self, mock_manager):
        """Test statistics with null values."""
        result = await get_statistics("test-session", columns=["numeric3", "mostly_null"])

        assert result.success is True
        numeric3_stats = next(s for s in result.statistics if s["column"] == "numeric3")
        assert numeric3_stats["null_count"] == 1
        assert numeric3_stats["null_percentage"] == 10.0

        mostly_null_stats = next(s for s in result.statistics if s["column"] == "mostly_null")
        assert mostly_null_stats["null_count"] == 7
        assert mostly_null_stats["null_percentage"] == 70.0

    async def test_statistics_non_numeric_columns(self, mock_manager):
        """Test statistics for non-numeric columns."""
        result = await get_statistics("test-session", columns=["categorical", "boolean"])

        assert result.success is True
        cat_stats = next(s for s in result.statistics if s["column"] == "categorical")
        assert cat_stats["unique"] == 3
        assert cat_stats["mode"] in ["A", "B", "C"]
        assert cat_stats["mean"] is None  # Non-numeric shouldn't have mean

    async def test_statistics_empty_dataframe(self, mock_manager):
        """Test statistics on empty dataframe."""
        mock_manager.return_value.get_session.return_value.data_session.df = pd.DataFrame()

        result = await get_statistics("test-session")
        assert result.success is True
        assert result.total_rows == 0
        assert result.total_columns == 0
        assert len(result.statistics) == 0

    async def test_statistics_invalid_columns(self, mock_manager):
        """Test statistics with invalid column names."""
        with pytest.raises(ToolError, match="not found"):
            await get_statistics("test-session", columns=["invalid_col"])

    async def test_statistics_mixed_valid_invalid_columns(self, mock_manager):
        """Test statistics with mix of valid and invalid columns."""
        with pytest.raises(ToolError, match="not found"):
            await get_statistics("test-session", columns=["numeric1", "invalid_col"])

    async def test_statistics_session_not_found(self):
        """Test statistics with invalid session."""
        with patch("src.databeak.servers.statistics_server.get_session_manager") as manager:
            manager.return_value.get_session.return_value = None

            with pytest.raises(ToolError, match="Session"):
                await get_statistics("invalid-session")

    async def test_statistics_no_data_loaded(self):
        """Test statistics when no data is loaded."""
        with patch("src.databeak.servers.statistics_server.get_session_manager") as manager:
            session = Mock()
            session.data_session.has_data.return_value = False
            manager.return_value.get_session.return_value = session

            with pytest.raises(ToolError, match="No data loaded"):
                await get_statistics("no-data-session")

    async def test_statistics_all_null_column(self, mock_manager):
        """Test statistics for column with all null values."""
        result = await get_statistics("test-session", columns=["all_null"])

        assert result.success is True
        null_stats = result.statistics[0]
        assert null_stats["null_count"] == 10
        assert null_stats["null_percentage"] == 100.0
        assert null_stats["mean"] is None


@pytest.mark.asyncio
class TestGetColumnStatistics:
    """Test get_column_statistics function."""

    async def test_column_statistics_numeric(self, mock_manager):
        """Test column statistics for numeric column."""
        result = await get_column_statistics("test-session", "numeric1")

        assert isinstance(result, ColumnStatisticsResult)
        assert result.success is True
        assert result.column == "numeric1"
        assert result.dtype == "int64"
        assert result.statistics["mean"] == 5.5
        assert result.statistics["q1"] == 3.25
        assert result.statistics["q3"] == 7.75

    async def test_column_statistics_categorical(self, mock_manager):
        """Test column statistics for categorical column."""
        result = await get_column_statistics("test-session", "categorical")

        assert result.success is True
        assert result.column == "categorical"
        assert result.statistics["unique"] == 3
        assert result.statistics["mode"] in ["A", "B", "C"]
        assert result.value_counts is not None
        assert len(result.value_counts) == 3

    async def test_column_statistics_datetime(self, mock_manager):
        """Test column statistics for datetime column."""
        result = await get_column_statistics("test-session", "dates")

        assert result.success is True
        assert result.column == "dates"
        assert "datetime" in str(result.dtype).lower()
        assert result.statistics["min"] is not None
        assert result.statistics["max"] is not None

    async def test_column_statistics_boolean(self, mock_manager):
        """Test column statistics for boolean column."""
        result = await get_column_statistics("test-session", "boolean")

        assert result.success is True
        assert result.column == "boolean"
        assert result.statistics["unique"] == 2
        assert result.value_counts is not None

    async def test_column_statistics_invalid_column(self, mock_manager):
        """Test column statistics with invalid column."""
        with pytest.raises(ToolError, match="Column"):
            await get_column_statistics("test-session", "invalid_column")

    async def test_column_statistics_with_nulls(self, mock_manager):
        """Test column statistics handling null values."""
        result = await get_column_statistics("test-session", "numeric3")

        assert result.success is True
        assert result.statistics["null_count"] == 1
        assert result.statistics["null_percentage"] == 10.0
        # Stats should be calculated excluding nulls
        assert result.statistics["count"] == 9


@pytest.mark.asyncio
class TestGetCorrelationMatrix:
    """Test get_correlation_matrix function."""

    async def test_correlation_matrix_default(self, mock_manager):
        """Test correlation matrix with default settings."""
        result = await get_correlation_matrix("test-session")

        assert isinstance(result, CorrelationResult)
        assert result.success is True
        assert result.matrix is not None
        assert len(result.columns) > 0
        assert result.method == "pearson"

        # Check matrix structure
        assert len(result.matrix) == len(result.columns)
        assert all(len(row) == len(result.columns) for row in result.matrix)

        # Check diagonal is 1.0
        for i in range(len(result.columns)):
            assert abs(result.matrix[i][i] - 1.0) < 0.001

    async def test_correlation_matrix_specific_columns(self, mock_manager):
        """Test correlation matrix for specific columns."""
        result = await get_correlation_matrix(
            "test-session", columns=["numeric1", "numeric2", "numeric3"]
        )

        assert result.success is True
        assert len(result.columns) == 3
        assert all(col in ["numeric1", "numeric2", "numeric3"] for col in result.columns)

    async def test_correlation_matrix_spearman(self, mock_manager):
        """Test correlation matrix with Spearman method."""
        result = await get_correlation_matrix("test-session", method="spearman")

        assert result.success is True
        assert result.method == "spearman"

    async def test_correlation_matrix_kendall(self, mock_manager):
        """Test correlation matrix with Kendall method."""
        result = await get_correlation_matrix("test-session", method="kendall")

        assert result.success is True
        assert result.method == "kendall"

    async def test_correlation_matrix_min_correlation(self, mock_manager):
        """Test correlation matrix with minimum correlation filter."""
        result = await get_correlation_matrix("test-session", min_correlation=0.5)

        assert result.success is True
        # Significant correlations should be identified
        assert result.significant_correlations is not None

    async def test_correlation_matrix_no_numeric_columns(self, mock_manager):
        """Test correlation matrix with no numeric columns."""
        mock_manager.return_value.get_session.return_value.data_session.df = pd.DataFrame(
            {"text1": ["a", "b", "c"], "text2": ["x", "y", "z"]}
        )

        result = await get_correlation_matrix("test-session")
        assert result.success is True
        assert len(result.columns) == 0
        assert len(result.matrix) == 0

    async def test_correlation_matrix_invalid_method(self, mock_manager):
        """Test correlation matrix with invalid method."""
        with pytest.raises(ToolError, match="method"):
            await get_correlation_matrix("test-session", method="invalid")

    async def test_correlation_matrix_invalid_columns(self, mock_manager):
        """Test correlation matrix with invalid columns."""
        with pytest.raises(ToolError, match="not found"):
            await get_correlation_matrix("test-session", columns=["numeric1", "invalid_col"])

    async def test_correlation_matrix_single_column(self, mock_manager):
        """Test correlation matrix with single column."""
        result = await get_correlation_matrix("test-session", columns=["numeric1"])

        assert result.success is True
        assert len(result.columns) == 1
        assert len(result.matrix) == 1
        assert result.matrix[0][0] == 1.0


@pytest.mark.asyncio
class TestGetValueCounts:
    """Test get_value_counts function."""

    async def test_value_counts_categorical(self, mock_manager):
        """Test value counts for categorical column."""
        result = await get_value_counts("test-session", "categorical")

        assert isinstance(result, ValueCountsResult)
        assert result.success is True
        assert result.column == "categorical"
        assert result.total_unique == 3
        assert len(result.counts) == 3

        # Check structure
        for count_item in result.counts:
            assert "value" in count_item
            assert "count" in count_item
            assert "percentage" in count_item

        # Check total percentage is 100
        total_percentage = sum(item["percentage"] for item in result.counts)
        assert abs(total_percentage - 100.0) < 0.01

    async def test_value_counts_numeric(self, mock_manager):
        """Test value counts for numeric column."""
        result = await get_value_counts("test-session", "numeric1")

        assert result.success is True
        assert result.total_unique == 10
        assert len(result.counts) == 10

    async def test_value_counts_with_nulls(self, mock_manager):
        """Test value counts with null values."""
        result = await get_value_counts("test-session", "mostly_null", dropna=False)

        assert result.success is True
        # Should include null as a value
        values = [item["value"] for item in result.counts]
        assert None in values or "NaN" in str(values)

    async def test_value_counts_dropna(self, mock_manager):
        """Test value counts dropping null values."""
        result = await get_value_counts("test-session", "mostly_null", dropna=True)

        assert result.success is True
        # Should not include null values
        values = [item["value"] for item in result.counts]
        assert None not in values

    async def test_value_counts_normalized(self, mock_manager):
        """Test normalized value counts."""
        result = await get_value_counts("test-session", "categorical", normalize=True)

        assert result.success is True
        # When normalized, counts should be proportions
        for count_item in result.counts:
            assert 0 <= count_item["count"] <= 1

    async def test_value_counts_top_n(self, mock_manager):
        """Test value counts with top N limit."""
        result = await get_value_counts("test-session", "categorical", top_n=2)

        assert result.success is True
        assert len(result.counts) <= 2

    async def test_value_counts_invalid_column(self, mock_manager):
        """Test value counts with invalid column."""
        with pytest.raises(ToolError, match="Column"):
            await get_value_counts("test-session", "invalid_column")

    async def test_value_counts_empty_column(self, mock_manager):
        """Test value counts for empty/all-null column."""
        result = await get_value_counts("test-session", "all_null", dropna=True)

        assert result.success is True
        assert result.total_unique == 0
        assert len(result.counts) == 0

    async def test_value_counts_datetime(self, mock_manager):
        """Test value counts for datetime column."""
        result = await get_value_counts("test-session", "dates")

        assert result.success is True
        assert result.total_unique == 10
        # Dates should be converted to strings
        assert all(isinstance(item["value"], str) for item in result.counts)


@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and error conditions."""

    async def test_large_dataframe(self, mock_manager):
        """Test with large dataframe."""
        large_df = pd.DataFrame({f"col_{i}": np.random.randn(10000) for i in range(100)})
        mock_manager.return_value.get_session.return_value.data_session.df = large_df

        result = await get_statistics("test-session")
        assert result.success is True
        assert result.total_rows == 10000
        assert result.total_columns == 100

    async def test_single_row_dataframe(self, mock_manager):
        """Test with single row dataframe."""
        single_row_df = pd.DataFrame({"col1": [1], "col2": ["text"], "col3": [True]})
        mock_manager.return_value.get_session.return_value.data_session.df = single_row_df

        result = await get_statistics("test-session")
        assert result.success is True
        assert result.total_rows == 1
        # Standard deviation should be NaN or 0 for single row
        stats = result.statistics[0]
        assert stats["std"] == 0 or pd.isna(stats["std"])

    async def test_all_same_values(self, mock_manager):
        """Test column with all same values."""
        same_values_df = pd.DataFrame({"constant": [5] * 10, "text_constant": ["same"] * 10})
        mock_manager.return_value.get_session.return_value.data_session.df = same_values_df

        result = await get_statistics("test-session")
        assert result.success is True

        const_stats = next(s for s in result.statistics if s["column"] == "constant")
        assert const_stats["std"] == 0
        assert const_stats["min"] == const_stats["max"]

        text_stats = next(s for s in result.statistics if s["column"] == "text_constant")
        assert text_stats["unique"] == 1

    async def test_extreme_values(self, mock_manager):
        """Test with extreme numeric values."""
        extreme_df = pd.DataFrame(
            {
                "tiny": [1e-100, 1e-99, 1e-98],
                "huge": [1e100, 1e101, 1e102],
                "mixed": [-1e50, 0, 1e50],
            }
        )
        mock_manager.return_value.get_session.return_value.data_session.df = extreme_df

        result = await get_statistics("test-session")
        assert result.success is True
        # Should handle extreme values without error

    async def test_special_characters_in_columns(self, mock_manager):
        """Test columns with special characters."""
        special_df = pd.DataFrame(
            {
                "column with spaces": [1, 2, 3],
                "column-with-dashes": [4, 5, 6],
                "column.with.dots": [7, 8, 9],
                "column@special#chars": [10, 11, 12],
            }
        )
        mock_manager.return_value.get_session.return_value.data_session.df = special_df

        result = await get_statistics("test-session")
        assert result.success is True
        assert result.total_columns == 4

    async def test_mixed_type_column_statistics(self, mock_manager):
        """Test statistics for mixed type column."""
        result = await get_column_statistics("test-session", "mixed")

        assert result.success is True
        assert result.column == "mixed"
        # Mixed types should still provide basic stats
        assert result.statistics["unique"] > 0
        assert result.statistics["null_count"] == 0
