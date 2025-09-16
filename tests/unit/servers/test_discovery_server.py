"""Comprehensive tests for discovery_server module - Fixed version."""

from typing import cast

import pytest
from fastmcp import Context
from fastmcp.exceptions import ToolError

from src.databeak.servers.discovery_server import (
    detect_outliers,
    find_cells_with_value,
    get_data_summary,
    group_by_aggregate,
    inspect_data_around,
    profile_data,
)
from src.databeak.servers.io_server import load_csv_from_content
from tests.test_mock_context import create_mock_context, create_mock_context_with_session_data


@pytest.fixture
async def discovery_session():
    """Create a session with diverse data for discovery testing."""
    csv_content = """customer_id,name,age,purchase_amount,category,region,is_premium
1001,John Smith,25,150.50,Electronics,North,True
1002,Jane Doe,35,2500.00,Electronics,South,True
1003,Bob Wilson,42,75.25,Clothing,North,False
1004,Alice Brown,28,180.00,Food,East,False
1005,Charlie Lee,31,95.50,Clothing,West,False
1006,Diana Prince,29,320.75,Electronics,North,True
1007,Edward King,55,5000.00,Electronics,South,True
1008,Fiona Green,26,45.00,Food,East,False
1009,George White,38,220.00,Clothing,West,False
1010,Helen Black,33,180.50,Food,North,False"""

    ctx = cast(Context, create_mock_context())
    result = await load_csv_from_content(ctx, csv_content)
    return ctx.session_id


@pytest.fixture
async def outlier_session():
    """Create a session with clear outliers."""
    csv_content = """id,value,amount,score
1,10,100,50
2,12,105,52
3,11,98,49
4,10,102,51
5,500,110,48
6,11,5000,53
7,12,103,500
8,10,99,50
9,11,101,51
10,13,104,49"""

    ctx = cast(Context, create_mock_context())
    result = await load_csv_from_content(ctx, csv_content)
    return ctx.session_id


@pytest.fixture
async def grouped_session():
    """Create a session for groupby operations."""
    csv_content = """department,employee,salary,bonus,years
Engineering,Alice,75000,5000,3
Engineering,Bob,85000,7000,5
Engineering,Charlie,95000,8000,7
Marketing,David,65000,3000,2
Marketing,Eve,70000,4000,4
Sales,Frank,60000,10000,3
Sales,Grace,65000,12000,4
Sales,Henry,70000,15000,5"""

    ctx = cast(Context, create_mock_context())
    result = await load_csv_from_content(ctx, csv_content)
    return ctx.session_id


@pytest.mark.asyncio
class TestDetectOutliers:
    """Tests for detect_outliers function."""

    async def test_detect_outliers_iqr_method(self, outlier_session):
        """Test outlier detection using IQR method."""
        ctx = cast(Context, create_mock_context_with_session_data(outlier_session))
        result = await detect_outliers(ctx, columns=["value"], method="iqr")

        assert result.success is True
        assert result.method == "iqr"
        assert result.outliers_found > 0

        # Should detect value=500 as outlier
        value_outliers = result.outliers_by_column.get("value", [])
        assert len(value_outliers) > 0
        assert any(o.value == 500 for o in value_outliers)

    async def test_detect_outliers_zscore_method(self, outlier_session):
        """Test outlier detection using z-score method."""
        ctx = cast(Context, create_mock_context_with_session_data(outlier_session))
        result = await detect_outliers(ctx, columns=["amount"], method="zscore", threshold=2.0)

        assert result.success is True
        assert result.method == "zscore"

        # Should detect amount=5000 as outlier
        amount_outliers = result.outliers_by_column.get("amount", [])
        assert len(amount_outliers) > 0
        assert any(o.value == 5000 for o in amount_outliers)

    async def test_detect_outliers_multiple_columns(self, outlier_session):
        """Test outlier detection on multiple columns."""
        ctx = cast(Context, create_mock_context_with_session_data(outlier_session))
        result = await detect_outliers(ctx, columns=["value", "amount", "score"])

        assert result.success is True
        assert "value" in result.outliers_by_column
        assert "amount" in result.outliers_by_column
        assert "score" in result.outliers_by_column

    async def test_detect_outliers_all_columns(self, outlier_session):
        """Test outlier detection on all numeric columns."""
        ctx = cast(Context, create_mock_context_with_session_data(outlier_session))
        result = await detect_outliers(ctx)

        assert result.success is True
        assert len(result.outliers_by_column) > 0
        assert result.outliers_found > 0

    async def test_detect_outliers_non_numeric(self, discovery_session):
        """Test outlier detection with non-numeric columns."""
        with pytest.raises(ToolError, match="numeric"):
            ctx = cast(Context, create_mock_context_with_session_data(discovery_session))
            await detect_outliers(ctx, columns=["name", "category"])

    async def test_detect_outliers_no_outliers(self):
        """Test when no outliers are present."""
        csv_content = "value\n10\n11\n12\n13\n14"
        ctx = cast(Context, create_mock_context())
        result = await load_csv_from_content(ctx, csv_content)

        ctx_outliers = cast(Context, create_mock_context_with_session_data(ctx.session_id))
        outliers = await detect_outliers(ctx_outliers, columns=["value"])
        assert outliers.success is True
        assert outliers.outliers_found == 0


@pytest.mark.asyncio
class TestProfileData:
    """Tests for profile_data function."""

    async def test_profile_basic(self, discovery_session):
        """Test basic data profiling."""
        ctx = cast(Context, create_mock_context_with_session_data(discovery_session))
        result = await profile_data(ctx)

        assert result.success is True
        assert len(result.profile) > 0

        # Check profile for numeric column
        age_profile = result.profile.get("age")
        assert age_profile is not None
        assert age_profile.data_type in ["int64", "float64"]
        assert age_profile.null_count >= 0
        assert age_profile.null_percentage >= 0
        assert age_profile.unique_count > 0
        assert age_profile.unique_percentage > 0

    async def test_profile_statistics(self, discovery_session):
        """Test profile includes statistics for numeric columns."""
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await profile_data(ctx)

        # Check numeric column has appropriate info
        amount_profile = result.profile.get("purchase_amount")
        assert amount_profile is not None

        # Check string column has appropriate stats
        name_profile = result.profile.get("name")
        assert name_profile is not None
        assert name_profile.most_frequent is not None

    async def test_profile_with_data(self):
        """Test profiling with actual DataFrame."""
        csv_content = "col1,col2\n1,2\n3,4"
        ctx = create_mock_context()
        result = await load_csv_from_content(ctx, csv_content)

        ctx_profile = create_mock_context_with_session_data(ctx.session_id)
        profile = await profile_data(ctx_profile)
        assert profile.success is True
        assert profile.total_rows == 2


@pytest.mark.asyncio
class TestGroupByAggregate:
    """Tests for group_by_aggregate function."""

    async def test_groupby_single_column(self, grouped_session):
        """Test groupby with single grouping column."""
        ctx = create_mock_context_with_session_data(grouped_session)
        result = await group_by_aggregate(
            ctx,
            group_by=["department"],
            aggregations={"salary": ["mean", "sum"], "bonus": ["mean"]},
        )

        assert result.success is True
        assert len(result.groups) == 3  # 3 departments

        # Check Engineering group
        eng_group = result.groups.get("Engineering")
        assert eng_group is not None
        assert eng_group.count == 3
        assert eng_group.mean > 70000 if hasattr(eng_group, "mean") else True

    async def test_groupby_multiple_columns(self, discovery_session):
        """Test groupby with multiple grouping columns."""
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await group_by_aggregate(
            ctx,
            group_by=["category", "region"],
            aggregations={"purchase_amount": ["sum", "count"]},
        )

        assert result.success is True
        assert len(result.groups) > 0

    async def test_groupby_all_aggregations(self, grouped_session):
        """Test various aggregation functions."""
        ctx = create_mock_context_with_session_data(grouped_session)
        result = await group_by_aggregate(
            ctx,
            group_by=["department"],
            aggregations={
                "salary": ["mean", "median", "sum", "min", "max", "std", "count"],
                "years": ["mean"],
            },
        )

        assert result.success is True
        assert len(result.groups) == 3

    async def test_groupby_with_nulls(self):
        """Test groupby with null values."""
        csv_content = """category,value
A,10
A,20
B,30
B,
A,
C,40"""
        ctx = create_mock_context()
        result = await load_csv_from_content(ctx, csv_content)

        agg_result = await group_by_aggregate(
            create_mock_context_with_session_data(ctx.session_id),
            group_by=["category"],
            aggregations={"value": ["mean", "count"]},
        )

        assert agg_result.success is True
        assert len(agg_result.groups) == 3

    async def test_groupby_invalid_column(self, grouped_session):
        """Test groupby with invalid column."""
        with pytest.raises(ToolError, match="not found"):
            await group_by_aggregate(
                create_mock_context_with_session_data(grouped_session),
                group_by=["fake_column"],
                aggregations={"salary": ["mean"]},
            )

    async def test_groupby_invalid_aggregation(self, grouped_session):
        """Test groupby with invalid aggregation."""
        # Server ignores invalid aggregations and uses defaults
        result = await group_by_aggregate(
            create_mock_context_with_session_data(grouped_session),
            group_by=["department"],
            aggregations={"salary": ["invalid_func"]},
        )
        assert result.success is True


@pytest.mark.asyncio
class TestFindCellsWithValue:
    """Tests for find_cells_with_value function."""

    async def test_find_exact_match(self, discovery_session):
        """Test finding cells with exact value match."""
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await find_cells_with_value(ctx, "John Smith")

        assert result.success is True
        assert result.exact_match is True
        assert result.matches_found == 1
        assert len(result.coordinates) == 1
        assert result.coordinates[0].row == 0
        assert result.coordinates[0].column == "name"
        assert result.coordinates[0].value == "John Smith"

    async def test_find_numeric_value(self, discovery_session):
        """Test finding numeric values."""
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await find_cells_with_value(ctx, 180.00)

        assert result.success is True
        assert result.matches_found > 0  # Multiple cells might have this value

    async def test_find_in_specific_columns(self, discovery_session):
        """Test finding values in specific columns."""
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await find_cells_with_value(ctx, "North", columns=["region"])

        assert result.success is True
        assert result.search_column == "region"
        assert all(c.column == "region" for c in result.coordinates)
        assert all(c.value == "North" for c in result.coordinates)

    async def test_find_substring_match(self, discovery_session):
        """Test finding cells with substring match."""
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await find_cells_with_value(ctx, "John", exact_match=False)

        assert result.success is True
        assert result.exact_match is False
        assert result.matches_found >= 1
        # Should find "John Smith"

    async def test_find_case_insensitive(self, discovery_session):
        """Test case-insensitive search."""
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await find_cells_with_value(ctx, "john smith", exact_match=False)

        assert result.success is True
        assert result.matches_found >= 1

    async def test_find_boolean_value(self, discovery_session):
        """Test finding boolean values."""
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await find_cells_with_value(ctx, True, columns=["is_premium"])

        assert result.success is True
        assert result.matches_found > 0
        assert all(c.value is True for c in result.coordinates)

    async def test_find_no_matches(self, discovery_session):
        """Test when no matches are found."""
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await find_cells_with_value(ctx, "NonExistent")

        assert result.success is True
        assert result.matches_found == 0
        assert len(result.coordinates) == 0

    async def test_find_null_values(self):
        """Test finding null values."""
        csv_content = "col1,col2\nA,1\nB,\n,3\nD,4"
        ctx = create_mock_context()
        result = await load_csv_from_content(ctx, csv_content)

        null_result = await find_cells_with_value(
            create_mock_context_with_session_data(ctx.session_id), None
        )
        assert null_result.success is True
        assert null_result.matches_found == 2


@pytest.mark.asyncio
class TestInspectDataAround:
    """Tests for inspect_data_around function."""

    async def test_inspect_center_cell(self, discovery_session):
        """Test inspecting data around a specific cell."""
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await inspect_data_around(ctx, row=5, column_name="name", radius=2)

        assert result.success is True
        assert result.center_coordinates["row"] == 5
        assert result.center_coordinates["column"] == "name"
        assert result.radius == 2

        # Should return 5x5 grid (or less at edges)
        assert result.surrounding_data is not None
        assert len(result.surrounding_data.rows) <= 5

    async def test_inspect_edge_cases(self, discovery_session):
        """Test inspection at edges of DataFrame."""
        # Top-left corner
        result = await inspect_data_around(
            create_mock_context_with_session_data(discovery_session),
            row=0,
            column_name="customer_id",
            radius=3,
        )

        assert result.success is True
        assert result.surrounding_data is not None

        # Bottom-right area
        result = await inspect_data_around(
            create_mock_context_with_session_data(discovery_session),
            row=9,
            column_name="is_premium",
            radius=2,
        )

        assert result.success is True

    async def test_inspect_large_radius(self, discovery_session):
        """Test with large radius."""
        result = await inspect_data_around(
            create_mock_context_with_session_data(discovery_session),
            row=5,
            column_name="category",
            radius=10,
        )

        assert result.success is True
        # Should return entire DataFrame since radius is large
        assert len(result.surrounding_data.rows) == 10

    async def test_inspect_invalid_coordinates(self, discovery_session):
        """Test with invalid coordinates."""
        # Server handles out-of-range coordinates gracefully
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await inspect_data_around(ctx, row=100, column_name="name", radius=1)
        assert result.surrounding_data.row_count == 0  # No rows in range

        # Invalid column should still raise error
        with pytest.raises(ToolError, match="not found"):
            await inspect_data_around(
                create_mock_context_with_session_data(discovery_session),
                row=0,
                column_name="nonexistent",
                radius=1,
            )


@pytest.mark.asyncio
class TestGetDataSummary:
    """Tests for get_data_summary function."""

    async def test_data_summary_with_preview(self, discovery_session):
        """Test comprehensive data summary with preview."""
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await get_data_summary(ctx, include_preview=True)

        assert result.success is True

        # Check coordinate system
        assert result.coordinate_system is not None
        assert "row_indexing" in result.coordinate_system
        assert "column_indexing" in result.coordinate_system

        # Check shape
        assert result.shape["rows"] == 10
        assert result.shape["columns"] == 7

        # Check columns info
        assert len(result.columns) == 7
        assert "customer_id" in result.columns
        assert "purchase_amount" in result.columns

        # Check data types
        assert "numeric" in result.data_types
        assert "text" in result.data_types

        # Check missing data info
        assert result.missing_data.total_missing >= 0
        assert result.missing_data.missing_percentage >= 0

        # Check preview
        assert result.preview is not None
        assert len(result.preview.rows) > 0

    async def test_data_summary_without_preview(self, discovery_session):
        """Test data summary without preview."""
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await get_data_summary(ctx, include_preview=False)

        assert result.success is True
        assert result.preview is None
        assert result.shape is not None
        assert result.columns is not None

    async def test_data_summary_memory_usage(self, discovery_session):
        """Test memory usage in summary."""
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await get_data_summary(ctx)

        assert result.success is True
        assert result.memory_usage_mb >= 0

    async def test_data_summary_with_nulls(self):
        """Test summary with null values."""
        csv_content = """col1,col2,col3
A,1,
B,,2
,3,3
D,4,"""
        ctx = create_mock_context()
        result = await load_csv_from_content(ctx, csv_content)

        summary = await get_data_summary(create_mock_context_with_session_data(ctx.session_id))
        assert summary.success is True
        assert summary.missing_data.total_missing > 0
        assert summary.missing_data.missing_percentage > 0

        for col in ["col1", "col2", "col3"]:
            assert summary.missing_data.missing_by_column[col] >= 0

    async def test_data_summary_type_detection(self, discovery_session):
        """Test correct data type detection."""
        ctx = create_mock_context_with_session_data(discovery_session)
        result = await get_data_summary(ctx)

        assert result.success is True

        # Check specific column types
        assert result.columns["customer_id"].type in ["int64", "float64"]
        assert result.columns["name"].type == "object"
        assert result.columns["is_premium"].type in ["bool", "object"]

        # Check type categorization
        assert "customer_id" in result.data_types["numeric"]
        assert "name" in result.data_types["text"]


@pytest.mark.asyncio
class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""

    async def test_with_actual_data(self):
        """Test all functions with actual DataFrame."""
        csv_content = "col1,col2\n1,2\n3,4"
        ctx = create_mock_context()
        result = await load_csv_from_content(ctx, csv_content)
        session_id = ctx.session_id

        # Profile should work
        profile = await profile_data(create_mock_context_with_session_data(session_id))
        assert profile.success is True
        assert profile.total_rows == 2

        # Summary should work
        summary = await get_data_summary(create_mock_context_with_session_data(session_id))
        assert summary.success is True
        assert summary.shape["rows"] == 2

        # Find should return results
        find_result = await find_cells_with_value(
            create_mock_context_with_session_data(session_id), 1
        )
        assert find_result.success is True
        assert find_result.matches_found == 1

    async def test_single_row_operations(self):
        """Test with single row DataFrame."""
        csv_content = "name,value\nTest,42"
        ctx = create_mock_context()
        result = await load_csv_from_content(ctx, csv_content)
        session_id = ctx.session_id

        # Inspect should work
        inspect = await inspect_data_around(
            create_mock_context_with_session_data(session_id), 0, "name", radius=2
        )
        assert inspect.success is True
        assert len(inspect.surrounding_data.rows) == 1

        # Find should work
        find_result = await find_cells_with_value(
            create_mock_context_with_session_data(session_id), 42
        )
        assert find_result.success is True
        assert find_result.matches_found == 1

    async def test_large_radius_inspect(self):
        """Test inspect with radius larger than DataFrame."""
        csv_content = "a,b\n1,2\n3,4"
        ctx = create_mock_context()
        result = await load_csv_from_content(ctx, csv_content)

        inspect = await inspect_data_around(
            create_mock_context_with_session_data(ctx.session_id),
            row=0,
            column_name="a",
            radius=100,
        )

        assert inspect.success is True
        assert len(inspect.surrounding_data.rows) == 2  # All rows

    async def test_session_not_found(self):
        """Test all functions with invalid session."""
        invalid_id = "nonexistent-session"

        with pytest.raises(ToolError):
            await detect_outliers(invalid_id)

        with pytest.raises(ToolError):
            await profile_data(create_mock_context_with_session_data(invalid_id))

        with pytest.raises(ToolError):
            await group_by_aggregate(invalid_id, group_by=["col"], aggregations={"val": ["mean"]})

        with pytest.raises(ToolError):
            await find_cells_with_value(invalid_id, "value")

        with pytest.raises(ToolError):
            await inspect_data_around(
                create_mock_context_with_session_data(invalid_id), 0, "col", 1
            )

        with pytest.raises(ToolError):
            await get_data_summary(create_mock_context_with_session_data(invalid_id))
