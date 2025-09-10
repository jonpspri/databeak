"""Comprehensive unit tests for discovery_server module to improve coverage."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

from src.databeak.servers.discovery_server import (
    DataSummaryResult,
    FindCellsResult,
    GroupAggregateResult,
    InspectDataResult,
    OutliersResult,
    ProfileResult,
    detect_outliers,
    find_cells_with_value,
    get_data_summary,
    group_by_aggregate,
    inspect_data_around,
    profile_data,
)


@pytest.fixture
def mock_session_with_outliers():
    """Create mock session with data containing outliers."""
    np.random.seed(42)
    normal_data = np.random.normal(50, 10, 95)  # Normal distribution
    outliers = [150, 200, -50, -100, 300]  # Clear outliers
    all_data = np.concatenate([normal_data, outliers])
    np.random.shuffle(all_data)

    session = Mock()
    session.data_session.df = pd.DataFrame(
        {
            "values": all_data,
            "values2": np.random.normal(100, 20, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "subcategory": np.random.choice(["X", "Y", "Z"], 100),
            "text": [f"item_{i}" for i in range(100)],
            "nulls": [None if i % 10 == 0 else i for i in range(100)],
            "dates": pd.date_range("2024-01-01", periods=100),
            "mixed": [i if i % 2 == 0 else f"text_{i}" for i in range(100)],
        }
    )
    session.data_session.has_data.return_value = True
    session.data_session.record_operation = Mock()
    return session


@pytest.fixture
def mock_manager(mock_session_with_outliers):
    """Create mock session manager."""
    with patch("src.databeak.servers.discovery_server.get_session_manager") as manager:
        manager.return_value.get_session.return_value = mock_session_with_outliers
        yield manager


@pytest.mark.asyncio
class TestDetectOutliers:
    """Test detect_outliers function comprehensively."""

    async def test_outliers_iqr_method(self, mock_manager):
        """Test outlier detection using IQR method."""
        result = await detect_outliers("test-session", columns=["values"], method="iqr")

        assert isinstance(result, OutliersResult)
        assert result.success is True
        assert result.method == "iqr"
        assert result.outliers_found > 0
        assert len(result.outlier_details) > 0

        # Check outlier details structure
        for detail in result.outlier_details:
            assert "column" in detail
            assert "row_index" in detail
            assert "value" in detail
            assert "reason" in detail

    async def test_outliers_zscore_method(self, mock_manager):
        """Test outlier detection using z-score method."""
        result = await detect_outliers(
            "test-session", columns=["values"], method="zscore", threshold=3.0
        )

        assert result.success is True
        assert result.method == "zscore"
        assert result.outliers_found > 0
        assert result.parameters["threshold"] == 3.0

    async def test_outliers_isolation_forest(self, mock_manager):
        """Test outlier detection using Isolation Forest."""
        result = await detect_outliers(
            "test-session",
            columns=["values", "values2"],
            method="isolation_forest",
            contamination=0.05,
        )

        assert result.success is True
        assert result.method == "isolation_forest"
        assert result.parameters["contamination"] == 0.05

    async def test_outliers_all_columns(self, mock_manager):
        """Test outlier detection on all numeric columns."""
        result = await detect_outliers("test-session", method="iqr")

        assert result.success is True
        # Should process all numeric columns
        column_names = {detail["column"] for detail in result.outlier_details}
        assert "values" in column_names

    async def test_outliers_invalid_method(self, mock_manager):
        """Test outlier detection with invalid method."""
        with pytest.raises(ToolError, match="Unknown outlier detection method"):
            await detect_outliers("test-session", method="invalid_method")

    async def test_outliers_non_numeric_columns(self, mock_manager):
        """Test outlier detection on non-numeric columns."""
        with pytest.raises(ToolError, match="not numeric"):
            await detect_outliers("test-session", columns=["text"])

    async def test_outliers_no_outliers_found(self, mock_manager):
        """Test when no outliers are found."""
        # Create data with no outliers
        mock_manager.return_value.get_session.return_value.data_session.df = pd.DataFrame(
            {
                "uniform": np.ones(100) * 50  # All same value
            }
        )

        result = await detect_outliers("test-session", columns=["uniform"], method="iqr")
        assert result.success is True
        assert result.outliers_found == 0
        assert len(result.outlier_details) == 0

    async def test_outliers_with_nulls(self, mock_manager):
        """Test outlier detection with null values."""
        result = await detect_outliers("test-session", columns=["nulls"], method="zscore")

        assert result.success is True
        # Should handle nulls gracefully

    async def test_outliers_custom_threshold(self, mock_manager):
        """Test outlier detection with custom threshold."""
        result = await detect_outliers(
            "test-session", columns=["values"], method="zscore", threshold=2.0
        )

        assert result.success is True
        assert result.parameters["threshold"] == 2.0

        # Lower threshold should find more outliers
        result2 = await detect_outliers(
            "test-session", columns=["values"], method="zscore", threshold=4.0
        )
        assert result.outliers_found >= result2.outliers_found


@pytest.mark.asyncio
class TestProfileData:
    """Test profile_data function."""

    async def test_profile_all_columns(self, mock_manager):
        """Test profiling all columns."""
        result = await profile_data("test-session")

        assert isinstance(result, ProfileResult)
        assert result.success is True
        assert result.total_rows == 100
        assert result.total_columns == 8
        assert len(result.column_profiles) == 8

        # Check column profile structure
        for profile in result.column_profiles:
            assert "column" in profile
            assert "dtype" in profile
            assert "null_count" in profile
            assert "null_percentage" in profile
            assert "unique_count" in profile
            assert "unique_percentage" in profile

    async def test_profile_specific_columns(self, mock_manager):
        """Test profiling specific columns."""
        result = await profile_data("test-session", columns=["values", "category", "text"])

        assert result.success is True
        assert len(result.column_profiles) == 3
        column_names = [p["column"] for p in result.column_profiles]
        assert set(column_names) == {"values", "category", "text"}

    async def test_profile_include_sample_values(self, mock_manager):
        """Test profile with sample values."""
        result = await profile_data(
            "test-session", columns=["category"], include_sample_values=True
        )

        assert result.success is True
        profile = result.column_profiles[0]
        assert "sample_values" in profile
        assert len(profile["sample_values"]) > 0

    async def test_profile_numeric_columns(self, mock_manager):
        """Test profiling numeric columns."""
        result = await profile_data("test-session", columns=["values"])

        profile = result.column_profiles[0]
        assert profile["dtype"] == "float64"
        assert "memory_usage" in profile
        assert "numeric_stats" in profile
        assert "mean" in profile["numeric_stats"]
        assert "std" in profile["numeric_stats"]
        assert "min" in profile["numeric_stats"]
        assert "max" in profile["numeric_stats"]

    async def test_profile_categorical_columns(self, mock_manager):
        """Test profiling categorical columns."""
        result = await profile_data("test-session", columns=["category"])

        profile = result.column_profiles[0]
        assert profile["dtype"] == "object"
        assert profile["unique_count"] == 3
        assert "most_frequent" in profile
        assert "most_frequent_count" in profile

    async def test_profile_datetime_columns(self, mock_manager):
        """Test profiling datetime columns."""
        result = await profile_data("test-session", columns=["dates"])

        profile = result.column_profiles[0]
        assert "datetime" in profile["dtype"]
        assert "date_range" in profile
        assert "min_date" in profile["date_range"]
        assert "max_date" in profile["date_range"]

    async def test_profile_mixed_type_columns(self, mock_manager):
        """Test profiling mixed type columns."""
        result = await profile_data("test-session", columns=["mixed"])

        profile = result.column_profiles[0]
        assert profile["dtype"] == "object"
        assert profile["unique_count"] > 0

    async def test_profile_quality_metrics(self, mock_manager):
        """Test profile with quality metrics."""
        result = await profile_data("test-session", include_quality_metrics=True)

        assert result.success is True
        assert "quality_metrics" in result.__dict__
        metrics = result.quality_metrics
        assert "completeness" in metrics
        assert "validity" in metrics

    async def test_profile_empty_dataframe(self, mock_manager):
        """Test profiling empty dataframe."""
        mock_manager.return_value.get_session.return_value.data_session.df = pd.DataFrame()

        result = await profile_data("test-session")
        assert result.success is True
        assert result.total_rows == 0
        assert result.total_columns == 0
        assert len(result.column_profiles) == 0


@pytest.mark.asyncio
class TestGroupByAggregate:
    """Test group_by_aggregate function."""

    async def test_group_by_single_column(self, mock_manager):
        """Test grouping by single column."""
        result = await group_by_aggregate(
            "test-session",
            group_columns=["category"],
            aggregations={"values": ["mean", "sum", "count"]},
        )

        assert isinstance(result, GroupAggregateResult)
        assert result.success is True
        assert result.group_columns == ["category"]
        assert len(result.groups) == 3  # A, B, C

        for group in result.groups:
            assert "category" in group
            assert "values_mean" in group
            assert "values_sum" in group
            assert "values_count" in group

    async def test_group_by_multiple_columns(self, mock_manager):
        """Test grouping by multiple columns."""
        result = await group_by_aggregate(
            "test-session",
            group_columns=["category", "subcategory"],
            aggregations={"values": ["mean"]},
        )

        assert result.success is True
        assert result.group_columns == ["category", "subcategory"]
        assert len(result.groups) > 0

    async def test_group_by_multiple_aggregations(self, mock_manager):
        """Test multiple aggregation functions."""
        result = await group_by_aggregate(
            "test-session",
            group_columns=["category"],
            aggregations={
                "values": ["mean", "median", "std", "min", "max"],
                "values2": ["sum", "count"],
            },
        )

        assert result.success is True
        group = result.groups[0]
        assert "values_mean" in group
        assert "values_median" in group
        assert "values_std" in group
        assert "values_min" in group
        assert "values_max" in group
        assert "values2_sum" in group
        assert "values2_count" in group

    async def test_group_by_invalid_column(self, mock_manager):
        """Test grouping by invalid column."""
        with pytest.raises(ToolError, match="not found"):
            await group_by_aggregate(
                "test-session", group_columns=["invalid_col"], aggregations={"values": ["mean"]}
            )

    async def test_group_by_invalid_aggregation(self, mock_manager):
        """Test invalid aggregation function."""
        with pytest.raises(ToolError, match="Unknown aggregation"):
            await group_by_aggregate(
                "test-session", group_columns=["category"], aggregations={"values": ["invalid_agg"]}
            )

    async def test_group_by_non_numeric_aggregation(self, mock_manager):
        """Test aggregating non-numeric columns."""
        result = await group_by_aggregate(
            "test-session", group_columns=["category"], aggregations={"text": ["count", "nunique"]}
        )

        assert result.success is True
        # Count and nunique should work for non-numeric

    async def test_group_by_with_nulls(self, mock_manager):
        """Test grouping with null values."""
        result = await group_by_aggregate(
            "test-session", group_columns=["category"], aggregations={"nulls": ["mean", "count"]}
        )

        assert result.success is True
        # Should handle nulls in aggregation


@pytest.mark.asyncio
class TestFindCellsWithValue:
    """Test find_cells_with_value function."""

    async def test_find_exact_match(self, mock_manager):
        """Test finding cells with exact match."""
        result = await find_cells_with_value("test-session", "A", columns=["category"])

        assert isinstance(result, FindCellsResult)
        assert result.success is True
        assert result.total_matches > 0
        assert len(result.locations) > 0

        for loc in result.locations:
            assert isinstance(loc, dict)
            assert "row" in loc
            assert "column" in loc
            assert "value" in loc
            assert loc["value"] == "A"

    async def test_find_numeric_value(self, mock_manager):
        """Test finding numeric value."""
        # Find a specific numeric value
        df = mock_manager.return_value.get_session.return_value.data_session.df
        target_value = df["values"].iloc[0]

        result = await find_cells_with_value("test-session", target_value, columns=["values"])

        assert result.success is True
        assert result.total_matches >= 1

    async def test_find_partial_match(self, mock_manager):
        """Test finding cells with partial match."""
        result = await find_cells_with_value(
            "test-session", "item", columns=["text"], exact_match=False
        )

        assert result.success is True
        assert result.total_matches == 100  # All text values contain "item"

    async def test_find_case_insensitive(self, mock_manager):
        """Test case-insensitive search."""
        result = await find_cells_with_value(
            "test-session", "a", columns=["category"], case_sensitive=False
        )

        assert result.success is True
        assert result.total_matches > 0

    async def test_find_in_all_columns(self, mock_manager):
        """Test finding value in all columns."""
        result = await find_cells_with_value("test-session", "A")

        assert result.success is True
        # Should search all columns

    async def test_find_null_values(self, mock_manager):
        """Test finding null values."""
        result = await find_cells_with_value("test-session", None, columns=["nulls"])

        assert result.success is True
        assert result.total_matches == 10  # Every 10th value is null

    async def test_find_no_matches(self, mock_manager):
        """Test when no matches are found."""
        result = await find_cells_with_value("test-session", "NONEXISTENT")

        assert result.success is True
        assert result.total_matches == 0
        assert len(result.locations) == 0

    async def test_find_max_results(self, mock_manager):
        """Test limiting maximum results."""
        result = await find_cells_with_value(
            "test-session", "A", columns=["category"], max_results=5
        )

        assert result.success is True
        assert len(result.locations) <= 5


@pytest.mark.asyncio
class TestGetDataSummary:
    """Test get_data_summary function."""

    async def test_data_summary_default(self, mock_manager):
        """Test getting default data summary."""
        result = await get_data_summary("test-session")

        assert isinstance(result, DataSummaryResult)
        assert result.success is True
        assert result.total_rows == 100
        assert result.total_columns == 8
        assert result.memory_usage_mb > 0

        # Check column types
        assert len(result.column_types) > 0
        assert "numeric" in result.column_types
        assert "object" in result.column_types

        # Check missing data
        assert isinstance(result.missing_data, dict)
        assert "total_missing" in result.missing_data
        assert "missing_by_column" in result.missing_data

        # Check preview
        assert result.preview is not None
        assert "columns" in result.preview
        assert "data" in result.preview

    async def test_data_summary_with_statistics(self, mock_manager):
        """Test data summary with basic statistics."""
        result = await get_data_summary("test-session", include_statistics=True)

        assert result.success is True
        assert "basic_stats" in result.__dict__
        assert result.basic_stats is not None

    async def test_data_summary_max_preview_rows(self, mock_manager):
        """Test data summary with custom preview size."""
        result = await get_data_summary("test-session", max_preview_rows=20)

        assert result.success is True
        assert len(result.preview["data"]) <= 20

    async def test_data_summary_empty_dataframe(self, mock_manager):
        """Test data summary for empty dataframe."""
        mock_manager.return_value.get_session.return_value.data_session.df = pd.DataFrame()

        result = await get_data_summary("test-session")
        assert result.success is True
        assert result.total_rows == 0
        assert result.total_columns == 0

    async def test_data_summary_large_dataframe(self, mock_manager):
        """Test data summary for large dataframe."""
        large_df = pd.DataFrame(np.random.randn(10000, 100))
        mock_manager.return_value.get_session.return_value.data_session.df = large_df

        result = await get_data_summary("test-session")
        assert result.success is True
        assert result.total_rows == 10000
        assert result.total_columns == 100
        assert result.memory_usage_mb > 0


@pytest.mark.asyncio
class TestInspectDataAround:
    """Test inspect_data_around function."""

    async def test_inspect_around_cell(self, mock_manager):
        """Test inspecting data around a specific cell."""
        result = await inspect_data_around("test-session", 50, "values", radius=5)

        assert isinstance(result, InspectDataResult)
        assert result.success is True
        assert result.center_row == 50
        assert result.center_column == "values"
        assert result.radius == 5

        # Check surrounding data
        assert "data" in result.surrounding_data
        assert len(result.surrounding_data["data"]) <= 11  # radius*2 + 1

    async def test_inspect_around_edge_cell(self, mock_manager):
        """Test inspecting around edge cells."""
        result = await inspect_data_around("test-session", 0, "values", radius=5)

        assert result.success is True
        assert result.center_row == 0
        # Should handle edge case gracefully
        assert len(result.surrounding_data["data"]) <= 6  # Can't go before row 0

    async def test_inspect_around_invalid_row(self, mock_manager):
        """Test inspecting around invalid row."""
        with pytest.raises(ToolError, match="Row index"):
            await inspect_data_around("test-session", 1000, "values")

    async def test_inspect_around_invalid_column(self, mock_manager):
        """Test inspecting around invalid column."""
        with pytest.raises(ToolError, match="Column"):
            await inspect_data_around("test-session", 50, "invalid_col")

    async def test_inspect_around_large_radius(self, mock_manager):
        """Test inspecting with large radius."""
        result = await inspect_data_around("test-session", 50, "values", radius=100)

        assert result.success is True
        # Should cap at dataframe boundaries
        assert len(result.surrounding_data["data"]) == 100

    async def test_inspect_around_with_context(self, mock_manager):
        """Test inspect with context information."""
        result = await inspect_data_around(
            "test-session", 50, "values", radius=3, include_column_context=True
        )

        assert result.success is True
        # Should include all columns in surrounding data
        assert len(result.surrounding_data["columns"]) == 8


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling across all functions."""

    async def test_session_not_found(self):
        """Test all functions with invalid session."""
        with patch("src.databeak.servers.discovery_server.get_session_manager") as manager:
            manager.return_value.get_session.return_value = None

            with pytest.raises(ToolError, match="Session"):
                await detect_outliers("invalid-session")

            with pytest.raises(ToolError, match="Session"):
                await profile_data("invalid-session")

            with pytest.raises(ToolError, match="Session"):
                await group_by_aggregate(
                    "invalid-session", group_columns=["col"], aggregations={"val": ["mean"]}
                )

    async def test_no_data_loaded(self):
        """Test all functions when no data is loaded."""
        with patch("src.databeak.servers.discovery_server.get_session_manager") as manager:
            session = Mock()
            session.data_session.has_data.return_value = False
            manager.return_value.get_session.return_value = session

            with pytest.raises(ToolError, match="No data"):
                await detect_outliers("no-data")

            with pytest.raises(ToolError, match="No data"):
                await profile_data("no-data")

    async def test_edge_cases(self, mock_manager):
        """Test various edge cases."""
        # Single row dataframe
        single_row_df = pd.DataFrame({"col": [1]})
        mock_manager.return_value.get_session.return_value.data_session.df = single_row_df

        result = await profile_data("test-session")
        assert result.success is True
        assert result.total_rows == 1

        # Single column dataframe
        single_col_df = pd.DataFrame({"only_col": range(100)})
        mock_manager.return_value.get_session.return_value.data_session.df = single_col_df

        result = await detect_outliers("test-session")
        assert result.success is True
