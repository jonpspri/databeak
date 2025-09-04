"""Additional tests for analytics module to improve coverage."""

import pytest
import pandas as pd

from src.csv_editor.tools.analytics import (
    detect_outliers,
    get_column_statistics,
    get_correlation_matrix,
    get_statistics,
    get_value_counts,
    group_by_aggregate,
    profile_data,
)
from src.csv_editor.tools.io_operations import load_csv_from_content


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
    return result["session_id"]


@pytest.mark.asyncio
class TestAnalyticsErrorHandling:
    """Test analytics error handling paths."""

    async def test_get_statistics_invalid_session(self):
        """Test statistics with invalid session."""
        result = await get_statistics("invalid-session")
        assert result["success"] is False
        assert "error" in result

    async def test_get_statistics_invalid_columns(self, analytics_test_session):
        """Test statistics with invalid column names."""
        result = await get_statistics(analytics_test_session, ["nonexistent_column"])
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_get_column_statistics_invalid_session(self):
        """Test column statistics with invalid session."""
        result = await get_column_statistics("invalid-session", "price")
        assert result["success"] is False
        assert "error" in result

    async def test_get_column_statistics_invalid_column(self, analytics_test_session):
        """Test column statistics with invalid column."""
        result = await get_column_statistics(analytics_test_session, "nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_detect_outliers_invalid_session(self):
        """Test outlier detection with invalid session."""
        result = await detect_outliers("invalid-session", ["price"])
        assert result["success"] is False
        assert "error" in result

    async def test_detect_outliers_invalid_columns(self, analytics_test_session):
        """Test outlier detection with invalid columns."""
        result = await detect_outliers(analytics_test_session, ["nonexistent"])
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_detect_outliers_non_numeric(self, analytics_test_session):
        """Test outlier detection on non-numeric columns."""
        result = await detect_outliers(analytics_test_session, ["category"])
        assert result["success"] is False
        assert "numeric" in result["error"]


@pytest.mark.asyncio
class TestAnalyticsAdvancedFeatures:
    """Test advanced analytics features."""

    async def test_get_correlation_matrix_success(self, analytics_test_session):
        """Test correlation matrix calculation."""
        result = await get_correlation_matrix(
            analytics_test_session, columns=["price", "quantity", "rating"]
        )
        assert result["success"] is True
        assert "correlation_matrix" in result
        assert len(result["correlation_matrix"]) == 3

    async def test_get_correlation_matrix_insufficient_columns(self, analytics_test_session):
        """Test correlation matrix with insufficient numeric columns."""
        result = await get_correlation_matrix(analytics_test_session, columns=["category"])
        assert result["success"] is False
        assert "numeric columns" in result["error"]

    async def test_group_by_aggregate_success(self, analytics_test_session):
        """Test group by aggregation."""
        result = await group_by_aggregate(
            analytics_test_session,
            group_by=["category"],
            aggregations={"price": ["mean", "max"], "quantity": ["sum"]},
        )
        assert result["success"] is True
        assert "grouped_data" in result
        assert "grouped_data" in result
        assert len(result["grouped_data"]) >= 2  # Should have category groups

    async def test_group_by_aggregate_invalid_columns(self, analytics_test_session):
        """Test group by with invalid columns."""
        result = await group_by_aggregate(
            analytics_test_session, group_by=["nonexistent"], aggregations={"price": ["mean"]}
        )
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_get_value_counts_success(self, analytics_test_session):
        """Test value counts functionality."""
        result = await get_value_counts(analytics_test_session, "category")
        assert result["success"] is True
        assert "value_counts" in result
        assert "Electronics" in result["value_counts"]
        assert "Furniture" in result["value_counts"]

    async def test_get_value_counts_invalid_column(self, analytics_test_session):
        """Test value counts with invalid column."""
        result = await get_value_counts(analytics_test_session, "nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_profile_data_success(self, analytics_test_session):
        """Test data profiling functionality."""
        result = await profile_data(analytics_test_session)
        assert result["success"] is True
        assert "profile" in result
        assert "columns" in result["profile"]
        assert len(result["profile"]["columns"]) == 5

    async def test_profile_data_invalid_session(self):
        """Test data profiling with invalid session."""
        result = await profile_data("invalid-session")
        assert result["success"] is False
        assert "error" in result


@pytest.mark.asyncio
class TestAnalyticsEdgeCases:
    """Test analytics edge cases and boundary conditions."""

    async def test_statistics_empty_dataframe(self):
        """Test statistics on empty dataframe."""
        # Create session with empty data
        result = await load_csv_from_content("name,age\n")  # Header only
        session_id = result["session_id"]

        # Test statistics
        stats_result = await get_statistics(session_id)
        assert stats_result["success"] is True
        assert stats_result["statistics"]["row_count"] == 0

    async def test_outliers_with_identical_values(self):
        """Test outlier detection with identical values."""
        # Create data with identical values (no outliers)
        csv_content = "value\n100\n100\n100\n100\n100"
        result = await load_csv_from_content(csv_content)
        session_id = result["session_id"]

        outliers_result = await detect_outliers(session_id, ["value"])
        assert outliers_result["success"] is True
        assert "outliers" in outliers_result
        # With identical values, there should be minimal outliers
        assert outliers_result["outliers"] is not None

    async def test_correlation_with_constant_column(self):
        """Test correlation with constant column."""
        # Create data where one column is constant
        csv_content = "var1,constant,var2\n1,100,10\n2,100,20\n3,100,30"
        result = await load_csv_from_content(csv_content)
        session_id = result["session_id"]

        corr_result = await get_correlation_matrix(session_id)
        assert corr_result["success"] is True
        # Constant column should have NaN correlations


@pytest.mark.asyncio
class TestAnalyticsDataTypes:
    """Test analytics with different data types."""

    async def test_statistics_mixed_types(self):
        """Test statistics on mixed data types."""
        csv_content = "text,number,boolean\nHello,123,True\nWorld,456,False"
        result = await load_csv_from_content(csv_content)
        session_id = result["session_id"]

        stats_result = await get_statistics(session_id)
        assert stats_result["success"] is True
        assert "statistics" in stats_result

    async def test_outliers_different_methods(self, analytics_test_session):
        """Test different outlier detection methods."""
        # Test IQR method
        result_iqr = await detect_outliers(analytics_test_session, ["price"], method="iqr")
        assert result_iqr["success"] is True

        # Test Z-score method
        result_zscore = await detect_outliers(analytics_test_session, ["price"], method="zscore")
        assert result_zscore["success"] is True

        # Test invalid method
        result_invalid = await detect_outliers(analytics_test_session, ["price"], method="invalid")
        assert result_invalid["success"] is False
        assert "method" in result_invalid["error"]
