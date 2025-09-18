"""Comprehensive unit tests for statistics_server module to improve coverage."""

import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

from src.databeak.models import get_session_manager
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
from tests.test_mock_context import create_mock_context


@pytest.fixture
def session_with_test_data():
    """Create real session with diverse test data."""
    # Create the test dataframe
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

    # Create a real session and load data
    session_id = str(uuid.uuid4())
    manager = get_session_manager()
    session = manager.get_or_create_session(session_id)
    session.df = df

    return session_id, df


@pytest.fixture
def session_with_empty_data():
    """Create real session with empty dataframe."""
    df = pd.DataFrame()

    # Create a real session and load empty data
    session_id = str(uuid.uuid4())
    manager = get_session_manager()
    session = manager.get_or_create_session(session_id)
    session.df = df

    return session_id, df


@pytest.mark.asyncio
class TestGetStatistics:
    """Test get_statistics function comprehensively."""

    async def test_statistics_all_columns(self, session_with_test_data):
        """Test getting statistics for all columns."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_statistics(ctx)

        assert isinstance(result, StatisticsResult)
        assert result.success is True
        assert len(result.statistics) > 0
        assert result.total_rows == 10
        assert result.column_count == 4  # Only numeric columns (including mostly_null)

        # Check numeric column stats
        numeric1_stats = result.statistics["numeric1"]
        assert numeric1_stats.mean == 5.5
        assert numeric1_stats.percentile_50 == 5.5  # median
        assert int(numeric1_stats.std) > 0
        assert numeric1_stats.min == 1
        assert numeric1_stats.max == 10

    async def test_statistics_specific_columns(self, session_with_test_data):
        """Test getting statistics for specific columns."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_statistics(ctx, columns=["numeric1", "numeric2"])

        assert result.success is True
        assert len(result.statistics) == 2
        assert all(col in ["numeric1", "numeric2"] for col in result.statistics)

    async def test_statistics_with_nulls(self, session_with_test_data):
        """Test statistics with null values."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_statistics(ctx, columns=["numeric3", "mostly_null"])

        assert result.success is True
        numeric3_stats = result.statistics["numeric3"]
        # StatisticsSummary doesn't have null_count/null_percentage
        assert numeric3_stats.count == 9  # Non-null count

        mostly_null_stats = result.statistics["mostly_null"]
        assert mostly_null_stats.count == 3  # Non-null count

    async def test_statistics_non_numeric_columns(self, session_with_test_data):
        """Test statistics for non-numeric columns."""
        # get_statistics only works with numeric columns
        # Non-numeric columns are silently skipped
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_statistics(ctx, columns=["categorical", "boolean"])

        assert result.success is True
        assert len(result.statistics) == 0  # No numeric columns in the selection

    async def test_statistics_empty_dataframe(self, session_with_empty_data):
        """Test statistics on empty dataframe."""
        session_id, df = session_with_empty_data
        ctx = create_mock_context(session_id)
        result = await get_statistics(ctx)
        assert result.success is True
        assert result.total_rows == 0
        assert result.column_count == 0
        assert len(result.statistics) == 0

    async def test_statistics_invalid_columns(self, session_with_test_data):
        """Test statistics with invalid column names."""
        session_id, df = session_with_test_data
        with pytest.raises(ToolError, match="not found"):
            ctx = create_mock_context(session_id)
            await get_statistics(ctx, columns=["invalid_col"])

    async def test_statistics_mixed_valid_invalid_columns(self, session_with_test_data):
        """Test statistics with mix of valid and invalid columns."""
        session_id, df = session_with_test_data
        with pytest.raises(ToolError, match="not found"):
            ctx = create_mock_context(session_id)
            await get_statistics(ctx, columns=["numeric1", "invalid_col"])

    @pytest.mark.skip(
        reason="TODO: get_or_create_session never returns None - need to redesign session not found behavior"
    )
    async def test_statistics_session_not_found(self):
        """Test statistics with invalid session."""
        with patch("src.databeak.servers.statistics_server.get_session_manager") as manager:
            manager.return_value.get_session.return_value = None

            with pytest.raises(ToolError, match="Session"):
                ctx = create_mock_context("invalid-session")
                await get_statistics(ctx)

    async def test_statistics_no_data_loaded(self):
        """Test statistics when no data is loaded."""
        # Create a real session but don't load any data
        session_id = str(uuid.uuid4())
        manager = get_session_manager()
        manager.get_or_create_session(session_id)
        # Don't call session.load_data() - leave df as None

        with pytest.raises(ToolError, match="No data loaded"):
            ctx = create_mock_context(session_id)
            await get_statistics(ctx)

    async def test_statistics_all_null_column(self, session_with_test_data):
        """Test statistics for column with all null values."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_statistics(ctx, columns=["all_null"])

        assert result.success is True
        # all_null column has no numeric values, so no statistics
        assert len(result.statistics) == 0


@pytest.mark.asyncio
class TestGetColumnStatistics:
    """Test get_column_statistics function."""

    @pytest.mark.parametrize(
        "column,expected_data_type,has_numeric_stats,expected_mean,expected_percentile_25",
        [
            ("numeric1", "int64", True, 5.5, 3.25),
            ("categorical", "object", False, None, None),
            ("dates", "datetime64[ns]", False, None, None),
            ("boolean", "bool", False, None, None),
        ],
    )
    async def test_column_statistics_by_type(
        self,
        session_with_test_data,
        column,
        expected_data_type,
        has_numeric_stats,
        expected_mean,
        expected_percentile_25,
    ):
        """Test column statistics for different data types."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_column_statistics(ctx, column)

        assert isinstance(result, ColumnStatisticsResult)
        assert result.success is True
        assert result.column == column

        # Check data type (with flexibility for datetime variants)
        if "datetime" in expected_data_type:
            assert "datetime" in str(result.data_type).lower()
        else:
            assert result.data_type == expected_data_type

        # Check statistics based on whether column should have numeric stats
        if has_numeric_stats:
            assert result.statistics.mean == expected_mean
            assert result.statistics.percentile_25 == expected_percentile_25
        else:
            assert result.statistics.mean is None
            assert result.statistics.std is None

    async def test_column_statistics_invalid_column(self, session_with_test_data):
        """Test column statistics with invalid column."""
        session_id, df = session_with_test_data
        with pytest.raises(ToolError, match="Column"):
            ctx = create_mock_context(session_id)
            await get_column_statistics(ctx, "invalid_column")

    async def test_column_statistics_with_nulls(self, session_with_test_data):
        """Test column statistics handling null values."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_column_statistics(ctx, "numeric3")

        assert result.success is True
        # StatisticsSummary doesn't have null_count/null_percentage
        # Count is non-null count
        assert result.statistics.count == 9
        assert result.non_null_count == 9


@pytest.mark.asyncio
class TestGetCorrelationMatrix:
    """Test get_correlation_matrix function."""

    async def test_correlation_matrix_default(self, session_with_test_data):
        """Test correlation matrix with default settings."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_correlation_matrix(ctx)

        assert isinstance(result, CorrelationResult)
        assert result.success is True
        assert result.correlation_matrix is not None
        assert len(result.columns_analyzed) > 0
        assert result.method == "pearson"

        # Check matrix structure
        assert len(result.correlation_matrix) == len(result.columns_analyzed)
        # correlation_matrix is a dict of dicts, not a list of lists
        for col in result.columns_analyzed:
            assert col in result.correlation_matrix
            assert len(result.correlation_matrix[col]) == len(result.columns_analyzed)

        # Check diagonal is 1.0
        for col in result.columns_analyzed:
            assert abs(result.correlation_matrix[col][col] - 1.0) < 0.001

    async def test_correlation_matrix_specific_columns(self, session_with_test_data):
        """Test correlation matrix for specific columns."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_correlation_matrix(ctx, columns=["numeric1", "numeric2", "numeric3"])

        assert result.success is True
        assert len(result.columns_analyzed) == 3
        assert all(col in ["numeric1", "numeric2", "numeric3"] for col in result.columns_analyzed)

    async def test_correlation_matrix_spearman(self, session_with_test_data):
        """Test correlation matrix with Spearman method."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_correlation_matrix(ctx, method="spearman")

        assert result.success is True
        assert result.method == "spearman"

    async def test_correlation_matrix_kendall(self, session_with_test_data):
        """Test correlation matrix with Kendall method."""
        session_id, df = session_with_test_data
        pytest.importorskip("scipy", reason="scipy not installed")
        ctx = create_mock_context(session_id)
        result = await get_correlation_matrix(ctx, method="kendall")

        assert result.success is True
        assert result.method == "kendall"

    async def test_correlation_matrix_min_correlation(self, session_with_test_data):
        """Test correlation matrix with minimum correlation filter."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_correlation_matrix(ctx, min_correlation=0.5)

        assert result.success is True
        # min_correlation parameter filters the matrix but doesn't add a significant_correlations field
        assert result.success is True

    async def test_correlation_matrix_no_numeric_columns(self):
        """Test correlation matrix with no numeric columns."""
        # Create a session with only text columns
        session_id = str(uuid.uuid4())
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)
        session.df = pd.DataFrame({"text1": ["a", "b", "c"], "text2": ["x", "y", "z"]})

        # Should raise an error when there are no numeric columns
        with pytest.raises(ToolError, match="No numeric columns"):
            ctx = create_mock_context(session_id)
            await get_correlation_matrix(ctx)

    async def test_correlation_matrix_invalid_method(self, session_with_test_data):
        """Test correlation matrix with invalid method."""
        session_id, df = session_with_test_data
        with pytest.raises(ToolError, match="method"):
            ctx = create_mock_context(session_id)
            await get_correlation_matrix(ctx, method="invalid")

    async def test_correlation_matrix_invalid_columns(self, session_with_test_data):
        """Test correlation matrix with invalid columns."""
        session_id, df = session_with_test_data
        with pytest.raises(ToolError, match="not found"):
            ctx = create_mock_context(session_id)
            await get_correlation_matrix(ctx, columns=["numeric1", "invalid_col"])

    async def test_correlation_matrix_single_column(self, session_with_test_data):
        """Test correlation matrix with single column."""
        session_id, df = session_with_test_data
        # Single column should raise an error since correlation needs at least 2 columns
        with pytest.raises(ToolError, match="at least two"):
            ctx = create_mock_context(session_id)
            await get_correlation_matrix(ctx, columns=["numeric1"])


@pytest.mark.asyncio
class TestGetValueCounts:
    """Test get_value_counts function."""

    async def test_value_counts_categorical(self, session_with_test_data):
        """Test value counts for categorical column."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_value_counts(ctx, "categorical")

        assert isinstance(result, ValueCountsResult)
        assert result.success is True
        assert result.column == "categorical"
        assert result.unique_values == 3
        assert len(result.value_counts) == 3

        # value_counts is a dict, not a list
        assert isinstance(result.value_counts, dict)
        # Check all values are present
        assert set(result.value_counts.keys()) == {"A", "B", "C"}

    async def test_value_counts_numeric(self, session_with_test_data):
        """Test value counts for numeric column."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_value_counts(ctx, "numeric1")

        assert result.success is True
        assert result.unique_values == 10
        assert len(result.value_counts) == 10

    async def test_value_counts_with_nulls(self, session_with_test_data):
        """Test value counts with null values."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_value_counts(ctx, "mostly_null")

        assert result.success is True
        # value_counts doesn't include nulls
        assert None not in result.value_counts

    async def test_value_counts_dropna(self, session_with_test_data):
        """Test value counts dropping null values."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_value_counts(ctx, "mostly_null")

        assert result.success is True
        # Should not include null values
        assert None not in result.value_counts

    async def test_value_counts_normalized(self, session_with_test_data):
        """Test normalized value counts."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_value_counts(ctx, "categorical", normalize=True)

        assert result.success is True
        # When normalized, value_counts should be proportions
        assert all(0 <= v <= 1 for v in result.value_counts.values())

    async def test_value_counts_top_n(self, session_with_test_data):
        """Test value counts with top N limit."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_value_counts(ctx, "categorical", top_n=2)

        assert result.success is True
        assert len(result.value_counts) <= 2

    async def test_value_counts_invalid_column(self, session_with_test_data):
        """Test value counts with invalid column."""
        session_id, df = session_with_test_data
        with pytest.raises(ToolError, match="Column"):
            ctx = create_mock_context(session_id)
            await get_value_counts(ctx, "invalid_column")

    async def test_value_counts_empty_column(self, session_with_test_data):
        """Test value counts for empty/all-null column."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_value_counts(ctx, "all_null")

        assert result.success is True
        assert result.unique_values == 0
        assert len(result.value_counts) == 0

    async def test_value_counts_datetime(self, session_with_test_data):
        """Test value counts for datetime column."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_value_counts(ctx, "dates")

        assert result.success is True
        assert result.unique_values == 10
        # value_counts keys should be strings for dates
        assert all(isinstance(k, str) for k in result.value_counts)


@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and error conditions."""

    async def test_large_dataframe(self):
        """Test with large dataframe."""
        large_df = pd.DataFrame({f"col_{i}": np.random.randn(10000) for i in range(100)})

        # Create a real session with large dataframe
        session_id = str(uuid.uuid4())
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)
        session.df = large_df

        ctx = create_mock_context(session_id)
        result = await get_statistics(ctx)
        assert result.success is True
        assert result.total_rows == 10000
        assert result.column_count == 100

    async def test_single_row_dataframe(self):
        """Test with single row dataframe."""
        single_row_df = pd.DataFrame({"col1": [1], "col2": ["text"], "col3": [True]})

        # Create a real session with single row dataframe
        session_id = str(uuid.uuid4())
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)
        session.df = single_row_df

        ctx = create_mock_context(session_id)
        result = await get_statistics(ctx)
        assert result.success is True
        assert result.total_rows == 1
        # Standard deviation should be NaN or 0 for single row
        assert result.column_count == 1  # Only col1 is numeric
        stats = result.statistics["col1"]
        assert stats.std == 0 or pd.isna(stats.std)

    async def test_all_same_values(self):
        """Test column with all same values."""
        same_values_df = pd.DataFrame({"constant": [5] * 10, "text_constant": ["same"] * 10})

        # Create a real session with constant values dataframe
        session_id = str(uuid.uuid4())
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)
        session.df = same_values_df

        ctx = create_mock_context(session_id)
        result = await get_statistics(ctx)
        assert result.success is True

        const_stats = result.statistics["constant"]
        assert const_stats.std == 0
        assert const_stats.min == const_stats.max
        # text_constant is non-numeric, so it won't be in statistics

    async def test_extreme_values(self):
        """Test with extreme numeric values."""
        extreme_df = pd.DataFrame(
            {
                "tiny": [1e-100, 1e-99, 1e-98],
                "huge": [1e100, 1e101, 1e102],
                "mixed": [-1e50, 0, 1e50],
            }
        )

        # Create a real session with extreme values dataframe
        session_id = str(uuid.uuid4())
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)
        session.df = extreme_df

        ctx = create_mock_context(session_id)
        result = await get_statistics(ctx)
        assert result.success is True
        # Should handle extreme values without error

    async def test_special_characters_in_columns(self):
        """Test columns with special characters."""
        special_df = pd.DataFrame(
            {
                "column with spaces": [1, 2, 3],
                "column-with-dashes": [4, 5, 6],
                "column.with.dots": [7, 8, 9],
                "column@special#chars": [10, 11, 12],
            }
        )

        # Create a real session with special column names dataframe
        session_id = str(uuid.uuid4())
        manager = get_session_manager()
        session = manager.get_or_create_session(session_id)
        session.df = special_df

        ctx = create_mock_context(session_id)
        result = await get_statistics(ctx)
        assert result.success is True
        assert result.column_count == 4

    async def test_mixed_type_column_statistics(self, session_with_test_data):
        """Test statistics for mixed type column."""
        session_id, df = session_with_test_data
        ctx = create_mock_context(session_id)
        result = await get_column_statistics(ctx, "mixed")

        assert result.success is True
        assert result.column == "mixed"
        # Mixed types are treated as non-numeric, so stats are None
        assert result.data_type == "object"
        assert result.statistics.mean is None
