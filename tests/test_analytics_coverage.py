"""Additional tests for analytics module to improve coverage."""

import pytest
from fastmcp.exceptions import ToolError

from src.databeak.tools.analytics import (
    detect_outliers,
    get_column_statistics,
    get_correlation_matrix,
    get_statistics,
    get_value_counts,
    group_by_aggregate,
    profile_data,
)
from src.databeak.tools.io_operations import load_csv_from_content


@pytest.fixture
async def analytics_test_session():
    """Create a session with analytics-friendly data."""
    csv_content = """product,price,quantity,category,rating
Laptop,999.99,10,Electronics,4.5
Mouse,29.99,50,Electronics,4.2
Keyboard,79.99,25,Electronics,4.0
Chair,299.99,15,Furniture,3.8
Desk,599.99,5,Furniture,4.1
Phone,799.99,20,Electronics,4.7
Table,399.99,8,Furniture,3.9"""

    result = await load_csv_from_content(csv_content)
    return result.session_id


@pytest.mark.asyncio
class TestAnalyticsErrorHandling:
    """Test analytics error handling paths."""

    async def test_get_statistics_invalid_session(self):
        """Test statistics with invalid session."""
        with pytest.raises(ToolError):
            await get_statistics("invalid-session")

    async def test_get_statistics_invalid_columns(self, analytics_test_session):
        """Test statistics with invalid column names."""
        with pytest.raises(ToolError, match="not found"):
            await get_statistics(analytics_test_session, ["nonexistent_column"])

    async def test_get_column_statistics_invalid_session(self):
        """Test column statistics with invalid session."""
        with pytest.raises(ToolError):
            await get_column_statistics("invalid-session", "price")

    async def test_get_column_statistics_invalid_column(self, analytics_test_session):
        """Test column statistics with invalid column."""
        with pytest.raises(ToolError, match="not found"):
            await get_column_statistics(analytics_test_session, "nonexistent")

    async def test_detect_outliers_invalid_session(self):
        """Test outlier detection with invalid session."""
        with pytest.raises(ToolError):
            await detect_outliers("invalid-session", ["price"])

    async def test_detect_outliers_invalid_columns(self, analytics_test_session):
        """Test outlier detection with invalid columns."""
        with pytest.raises(ToolError, match="not found"):
            await detect_outliers(analytics_test_session, ["nonexistent"])

    async def test_detect_outliers_non_numeric(self, analytics_test_session):
        """Test outlier detection on non-numeric columns."""
        with pytest.raises(ToolError, match="numeric"):
            await detect_outliers(analytics_test_session, ["category"])


@pytest.mark.asyncio
class TestGetColumnStatistics:
    """Comprehensive tests for get_column_statistics function."""

    async def test_get_column_statistics_numeric_success(self, analytics_test_session):
        """Test column statistics with numeric column."""
        result = await get_column_statistics(analytics_test_session, "price")
        assert result.success is True
        assert result.session_id == analytics_test_session
        assert result.column == "price"
        assert result.data_type == "float64"

        # Verify statistics structure
        assert hasattr(result, "statistics")
        assert result.statistics.count > 0
        assert result.statistics.mean > 0
        assert result.statistics.std >= 0
        assert result.statistics.min <= result.statistics.max
        assert (
            result.statistics.percentile_25
            <= result.statistics.percentile_50
            <= result.statistics.percentile_75
        )

        # Verify non_null_count makes sense
        assert result.non_null_count > 0
        assert result.non_null_count == result.statistics.count

    async def test_get_column_statistics_integer_column(self, analytics_test_session):
        """Test column statistics with integer column."""
        result = await get_column_statistics(analytics_test_session, "quantity")
        assert result.success is True
        assert result.column == "quantity"
        assert result.data_type == "int64"

        # Verify statistics are numeric and reasonable
        assert result.statistics.count > 0
        assert isinstance(result.statistics.mean, float)
        assert isinstance(result.statistics.min, float)
        assert isinstance(result.statistics.max, float)

    async def test_get_column_statistics_string_column(self, analytics_test_session):
        """Test column statistics with string/object column."""
        result = await get_column_statistics(analytics_test_session, "category")
        assert result.success is True
        assert result.column == "category"
        assert result.data_type == "object"

        # For non-numeric columns, statistics should be placeholder values
        assert result.statistics.count > 0  # Should count non-null values
        assert result.statistics.mean == 0.0
        assert result.statistics.std == 0.0
        assert result.statistics.min == 0.0
        assert result.statistics.max == 0.0

    async def test_get_column_statistics_with_nulls(self):
        """Test column statistics with null values."""
        csv_content = "values\n10\n20\n\n30\n40\n"  # One null value
        result = await load_csv_from_content(csv_content)
        session_id = result.session_id

        stats_result = await get_column_statistics(session_id, "values")
        assert stats_result.success is True
        assert stats_result.statistics.count == 4  # Should exclude null
        assert stats_result.non_null_count == 4

        # Verify statistics are calculated correctly for non-null values
        assert stats_result.statistics.mean == 25.0  # (10+20+30+40)/4
        assert stats_result.statistics.min == 10.0
        assert stats_result.statistics.max == 40.0

    async def test_get_column_statistics_single_value(self):
        """Test column statistics with single value."""
        csv_content = "single\n42"
        result = await load_csv_from_content(csv_content)
        session_id = result.session_id

        stats_result = await get_column_statistics(session_id, "single")
        assert stats_result.success is True
        assert stats_result.statistics.count == 1
        assert stats_result.statistics.mean == 42.0
        assert stats_result.statistics.min == 42.0
        assert stats_result.statistics.max == 42.0
        # Single value std is NaN in pandas, which is mathematically correct
        import math

        assert math.isnan(stats_result.statistics.std)

    async def test_get_column_statistics_empty_numeric_column(self):
        """Test column statistics with all null numeric values."""
        csv_content = "empty\n\n\n\n"  # All null values
        result = await load_csv_from_content(csv_content)
        session_id = result.session_id

        stats_result = await get_column_statistics(session_id, "empty")
        assert stats_result.success is True
        assert stats_result.statistics.count == 0
        assert stats_result.non_null_count == 0
        assert stats_result.statistics.mean == 0.0

    async def test_get_column_statistics_data_accuracy(self, analytics_test_session):
        """Test accuracy of calculated statistics against known data."""
        # Use 'rating' column which should have known values: [4.5, 4.2, 4.0, 3.8, 4.1, 4.7, 3.9]
        result = await get_column_statistics(analytics_test_session, "rating")
        assert result.success is True

        # Verify basic statistical properties
        assert result.statistics.count == 7
        assert 3.8 <= result.statistics.min <= 4.0  # Should be around 3.8
        assert 4.5 <= result.statistics.max <= 4.7  # Should be around 4.7
        assert 4.0 <= result.statistics.mean <= 4.5  # Should be reasonable average
        assert result.statistics.std > 0  # Should have some variation

    async def test_get_column_statistics_boolean_column(self):
        """Test column statistics with boolean column."""
        csv_content = "flag\nTrue\nFalse\nTrue\nFalse"
        result = await load_csv_from_content(csv_content)
        session_id = result.session_id

        stats_result = await get_column_statistics(session_id, "flag")
        assert stats_result.success is True
        assert stats_result.column == "flag"
        assert stats_result.data_type == "bool"

        # Boolean columns should get placeholder statistics (not numeric calculations)
        assert stats_result.statistics.count > 0  # Should count non-null values
        assert stats_result.statistics.mean == 0.0
        assert stats_result.statistics.std == 0.0


@pytest.mark.asyncio
class TestAnalyticsAdvancedFeatures:
    """Test advanced analytics features."""

    async def test_get_correlation_matrix_success(self, analytics_test_session):
        """Test correlation matrix calculation."""
        result = await get_correlation_matrix(
            analytics_test_session, columns=["price", "quantity", "rating"]
        )
        assert result.success is True
        assert hasattr(result, "correlation_matrix")
        assert len(result.correlation_matrix) == 3

    async def test_get_correlation_matrix_insufficient_columns(self, analytics_test_session):
        """Test correlation matrix with insufficient numeric columns."""
        with pytest.raises(ToolError, match="numeric columns"):
            await get_correlation_matrix(analytics_test_session, columns=["category"])

    async def test_group_by_aggregate_success(self, analytics_test_session):
        """Test group by aggregation."""
        result = await group_by_aggregate(
            analytics_test_session,
            group_by=["category"],
            aggregations={"price": ["mean", "max"], "quantity": ["sum"]},
        )
        assert result.success is True
        assert hasattr(result, "groups")
        assert hasattr(result, "groups")
        assert len(result.groups) >= 2  # Should have category groups

    async def test_group_by_aggregate_invalid_columns(self, analytics_test_session):
        """Test group by with invalid columns."""
        with pytest.raises(ToolError, match="not found"):
            await group_by_aggregate(
                analytics_test_session,
                group_by=["nonexistent"],
                aggregations={"price": ["mean"]},
            )

    async def test_get_value_counts_success(self, analytics_test_session):
        """Test value counts functionality."""
        result = await get_value_counts(analytics_test_session, "category")
        assert result.success is True
        assert hasattr(result, "value_counts")
        assert "Electronics" in result.value_counts
        assert "Furniture" in result.value_counts

    async def test_get_value_counts_invalid_column(self, analytics_test_session):
        """Test value counts with invalid column."""
        with pytest.raises(ToolError, match="not found"):
            await get_value_counts(analytics_test_session, "nonexistent")

    async def test_profile_data_success(self, analytics_test_session):
        """Test data profiling functionality."""
        result = await profile_data(analytics_test_session)
        assert result.success is True
        assert hasattr(result, "profile")
        # Profile is a dict mapping column names to ProfileInfo objects
        assert isinstance(result.profile, dict)
        assert len(result.profile) == 5

    async def test_profile_data_invalid_session(self):
        """Test data profiling with invalid session."""
        with pytest.raises(ToolError):
            await profile_data("invalid-session")


@pytest.mark.asyncio
class TestAnalyticsEdgeCases:
    """Test analytics edge cases and boundary conditions."""

    async def test_statistics_empty_dataframe(self):
        """Test statistics on empty dataframe."""
        # Create session with empty data
        result = await load_csv_from_content("name,age\n")  # Header only
        session_id = result.session_id

        # Test statistics
        stats_result = await get_statistics(session_id)
        assert stats_result.success is True
        # Statistics for empty dataframe should have total_rows or count of 0
        assert hasattr(stats_result, "statistics")
        assert (
            stats_result.statistics.get("total_rows", 0) == 0
            or stats_result.statistics.get("count", 0) == 0
        )

    async def test_outliers_with_identical_values(self):
        """Test outlier detection with identical values."""
        # Create data with identical values (no outliers)
        csv_content = "value\n100\n100\n100\n100\n100"
        result = await load_csv_from_content(csv_content)
        session_id = result.session_id

        outliers_result = await detect_outliers(session_id, ["value"])
        assert outliers_result.success is True
        assert hasattr(outliers_result, "outliers_by_column")
        # With identical values, there should be minimal outliers
        assert outliers_result.outliers_by_column is not None
        assert outliers_result.outliers_found == 0  # Identical values should have no outliers

    async def test_correlation_with_constant_column(self):
        """Test correlation with constant column."""
        # Create data where one column is constant
        csv_content = "var1,constant,var2\n1,100,10\n2,100,20\n3,100,30"
        result = await load_csv_from_content(csv_content)
        session_id = result.session_id

        corr_result = await get_correlation_matrix(session_id)
        assert corr_result.success is True
        # Constant column should have NaN correlations


@pytest.mark.asyncio
class TestAnalyticsDataTypes:
    """Test analytics with different data types."""

    async def test_statistics_mixed_types(self):
        """Test statistics on mixed data types."""
        csv_content = "text,number,boolean\nHello,123,True\nWorld,456,False"
        result = await load_csv_from_content(csv_content)
        session_id = result.session_id

        stats_result = await get_statistics(session_id)
        assert stats_result.success is True
        assert hasattr(stats_result, "statistics")

    async def test_outliers_different_methods(self, analytics_test_session):
        """Test different outlier detection methods."""
        # Test IQR method
        result_iqr = await detect_outliers(analytics_test_session, ["price"], method="iqr")
        assert result_iqr.success is True

        # Test Z-score method
        result_zscore = await detect_outliers(analytics_test_session, ["price"], method="zscore")
        assert result_zscore.success is True

        # Test invalid method
        with pytest.raises(ToolError, match="method"):
            await detect_outliers(analytics_test_session, ["price"], method="invalid")
