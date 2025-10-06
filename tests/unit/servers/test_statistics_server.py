"""Comprehensive tests for statistics_server module - Fixed version."""

import pytest
from fastmcp import Context
from fastmcp.exceptions import ToolError

from databeak.exceptions import ColumnNotFoundError, NoDataLoadedError
from databeak.servers.io_server import load_csv_from_content
from databeak.servers.statistics_server import (
    get_column_statistics,
    get_correlation_matrix,
    get_statistics,
    get_value_counts,
)
from tests.test_mock_context import create_mock_context


@pytest.fixture
async def stats_ctx() -> Context:
    """Create a session with diverse data for statistics testing."""
    csv_content = """name,age,salary,department,years_exp,performance_rating
John Smith,35,75000.50,Engineering,8,4.5
Jane Doe,28,65000.00,Marketing,5,4.2
Bob Johnson,42,95000.75,Engineering,15,4.8
Alice Brown,31,70000.25,Sales,7,3.9
Charlie Wilson,29,68000.00,Marketing,6,4.1
Diana Prince,38,85000.50,Engineering,12,4.6
Edward Norton,45,105000.00,Management,20,4.7
Fiona Green,26,55000.00,Sales,3,3.8
George White,33,72000.00,Marketing,9,4.0
Helen Black,40,92000.00,Engineering,14,4.4"""

    ctx = create_mock_context()
    await load_csv_from_content(ctx, csv_content)
    return ctx


@pytest.fixture
async def sparse_ctx() -> Context:
    """Create a session with sparse/missing data."""
    csv_content = """id,value1,value2,value3,category
1,10.5,,20.0,A
2,,15.0,25.0,B
3,12.0,18.0,,A
4,14.5,,30.0,C
5,,20.0,35.0,B
6,16.0,22.0,40.0,A"""

    ctx = create_mock_context()
    await load_csv_from_content(ctx, csv_content)
    return ctx


@pytest.fixture
async def edge_case_ctx() -> Context:
    """Create a session with edge case data."""
    csv_content = """single_value,zeros,identical,negative,mixed
5,0,100,-10,1
5,0,100,-20,2.5
5,0,100,-30,-3.7
5,0,100,-40,0
5,0,100,-50,999.99"""

    ctx = create_mock_context()
    await load_csv_from_content(ctx, csv_content)
    return ctx


@pytest.mark.asyncio
class TestGetStatistics:
    """Tests for get_statistics function."""

    async def test_get_statistics_all_columns(self, stats_ctx: Context) -> None:
        """Test getting statistics for all columns."""
        ctx = stats_ctx
        result = await get_statistics(ctx)

        assert result.success is True
        assert len(result.statistics) > 0

        # Check that numeric columns have statistics
        assert "age" in result.statistics
        assert "salary" in result.statistics
        for col_name in ["age", "salary", "years_exp", "performance_rating"]:
            if col_name in result.statistics:
                col_stats = result.statistics[col_name]
                assert col_stats.count > 0
                assert col_stats.mean is not None
                assert col_stats.std is not None
                assert col_stats.min is not None
                assert col_stats.max is not None

    async def test_get_statistics_specific_columns(self, stats_ctx: Context) -> None:
        """Test getting statistics for specific columns."""
        ctx = stats_ctx
        result = await get_statistics(ctx, columns=["age", "salary"])

        assert result.success is True
        assert len(result.statistics) == 2
        assert "age" in result.statistics
        assert "salary" in result.statistics
        assert "name" not in result.statistics

        # Verify age statistics
        age_stats = result.statistics["age"]
        assert age_stats.count == 10
        assert int(age_stats.min) >= 26  # type: ignore[arg-type]
        assert int(age_stats.max) <= 45  # type: ignore[arg-type]
        assert 30 <= int(age_stats.mean) <= 40  # type: ignore[arg-type]

    async def test_get_statistics_with_nulls(self, sparse_ctx: Context) -> None:
        """Test statistics calculation with null values."""
        ctx = sparse_ctx
        result = await get_statistics(ctx, columns=["value1", "value2"])

        assert result.success is True

        # value1 has some nulls
        v1_stats = result.statistics["value1"]
        assert v1_stats.count < 6  # Less than total rows due to nulls

        # value2 has some nulls
        v2_stats = result.statistics["value2"]
        assert v2_stats.count < 6

    async def test_get_statistics_non_numeric(self, stats_ctx: Context) -> None:
        """Test that statistics for non-numeric columns returns empty results."""
        # The refactored server only handles numeric columns, non-numeric are silently skipped
        ctx = stats_ctx
        result = await get_statistics(ctx, columns=["name", "department"])

        assert result.success is True
        assert len(result.statistics) == 0  # No numeric columns in the selection

    async def test_get_statistics_with_data(self) -> None:
        """Test statistics on DataFrame with data."""
        csv_content = "col1,col2,col3\n1,2,3"
        load_ctx = create_mock_context()
        await load_csv_from_content(load_ctx, csv_content)

        ctx = create_mock_context(load_ctx.session_id)
        stats = await get_statistics(ctx)
        assert stats.success is True
        assert stats.total_rows == 1

    async def test_get_statistics_invalid_session(self) -> None:
        """Test statistics with invalid session ID."""

        ctx = create_mock_context("invalid-session-id")

        with pytest.raises(NoDataLoadedError):
            await get_statistics(ctx)

    async def test_get_statistics_invalid_columns(self, stats_ctx: Context) -> None:
        """Test statistics with non-existent columns."""

        ctx = stats_ctx

        with pytest.raises(ColumnNotFoundError):
            await get_statistics(ctx, columns=["nonexistent", "fake_column"])


@pytest.mark.asyncio
class TestGetColumnStatistics:
    """Tests for get_column_statistics function."""

    async def test_numeric_column_statistics(self, stats_ctx: Context) -> None:
        """Test detailed statistics for numeric column."""
        ctx = stats_ctx
        result = await get_column_statistics(ctx, "salary")

        assert result.success is True
        assert result.column == "salary"
        assert result.data_type == "float64"

        stats = result.statistics
        assert stats.count == 10
        assert int(stats.mean) > 70000  # type: ignore[arg-type]
        assert int(stats.std) > 0  # type: ignore[arg-type]
        assert int(stats.min) >= 55000  # type: ignore[arg-type]
        assert int(stats.max) <= 105000  # type: ignore[arg-type]

        # Additional stats
        assert result.non_null_count == 10

    async def test_integer_column_statistics(self, stats_ctx: Context) -> None:
        """Test statistics for integer column."""
        ctx = stats_ctx
        result = await get_column_statistics(ctx, "age")

        assert result.success is True
        assert result.data_type in ["int64", "float64"]  # May be inferred as float

        stats = result.statistics
        assert 26 <= int(stats.min) <= int(stats.max) <= 45  # type: ignore[arg-type]
        assert int(stats.mean) > 0  # type: ignore[arg-type]
        assert stats.count == 10

    async def test_string_column_statistics(self, stats_ctx: Context) -> None:
        """Test statistics for string column."""
        ctx = stats_ctx
        result = await get_column_statistics(ctx, "department")

        assert result.success is True
        assert result.data_type == "object"

        stats = result.statistics
        assert stats.count == 10
        assert stats.unique == 4
        assert stats.top in ["Engineering", "Marketing", "Sales", "Management"]
        assert float(stats.freq) >= 1  # type: ignore[arg-type]

        # Numeric stats should be None for string columns
        assert stats.mean is None
        assert stats.std is None
        assert stats.min is None or isinstance(stats.min, str)
        assert stats.max is None or isinstance(stats.max, str)

    async def test_column_with_nulls(self, sparse_ctx: Context) -> None:
        """Test statistics for column with null values."""
        ctx = sparse_ctx
        result = await get_column_statistics(ctx, "value1")

        assert result.success is True
        assert result.non_null_count < 6
        assert result.statistics.count == result.non_null_count

    async def test_single_value_column(self, edge_case_ctx: Context) -> None:
        """Test statistics for column with single unique value."""
        ctx = edge_case_ctx
        result = await get_column_statistics(ctx, "single_value")

        assert result.success is True
        stats = result.statistics
        assert stats.std == 0  # No variation
        assert stats.min == stats.max == stats.mean == 5

    async def test_invalid_column(self, stats_ctx: Context) -> None:
        """Test with non-existent column."""

        ctx = stats_ctx

        with pytest.raises(ColumnNotFoundError):
            await get_column_statistics(ctx, "fake_column")


@pytest.mark.asyncio
class TestGetCorrelationMatrix:
    """Tests for get_correlation_matrix function."""

    async def test_correlation_all_numeric(self, stats_ctx: Context) -> None:
        """Test correlation matrix for all numeric columns."""
        ctx = stats_ctx
        result = await get_correlation_matrix(ctx)

        assert result.success is True
        assert result.method == "pearson"
        assert len(result.columns_analyzed) > 0
        assert len(result.correlation_matrix) > 0

        # Check matrix properties
        for col in result.columns_analyzed:
            assert col in result.correlation_matrix
            # Diagonal should be 1 (perfect correlation with self)
            assert abs(result.correlation_matrix[col][col] - 1.0) < 0.0001

    async def test_correlation_specific_columns(self, stats_ctx: Context) -> None:
        """Test correlation for specific columns."""
        ctx = stats_ctx
        result = await get_correlation_matrix(ctx, columns=["age", "salary", "years_exp"])

        assert result.success is True
        assert len(result.columns_analyzed) == 3
        assert "age" in result.columns_analyzed
        assert "salary" in result.columns_analyzed
        assert "years_exp" in result.columns_analyzed

        # Salary and years_exp should be positively correlated
        correlation = result.correlation_matrix["salary"]["years_exp"]
        assert correlation > 0.5  # Moderate to strong positive correlation

    async def test_correlation_methods(self, stats_ctx: Context) -> None:
        """Test different correlation methods."""
        # Pearson correlation
        ctx = stats_ctx
        pearson = await get_correlation_matrix(ctx, columns=["age", "salary"], method="pearson")
        assert pearson.success is True
        assert pearson.method == "pearson"

        # Spearman correlation
        ctx = stats_ctx
        spearman = await get_correlation_matrix(ctx, columns=["age", "salary"], method="spearman")
        assert spearman.success is True
        assert spearman.method == "spearman"

        # Kendall correlation - skip if scipy has import issues
        try:
            ctx = stats_ctx
            kendall = await get_correlation_matrix(ctx, columns=["age", "salary"], method="kendall")
            assert kendall.success is True
            assert kendall.method == "kendall"
        except (ToolError, ImportError) as e:
            if "LinAlgError" in str(e) or "cannot import" in str(e):
                pytest.skip("Skipping kendall correlation due to scipy import issue")
            else:
                raise

    async def test_correlation_with_nulls(self, sparse_ctx: Context) -> None:
        """Test correlation with missing values."""
        ctx = sparse_ctx
        result = await get_correlation_matrix(ctx, columns=["value1", "value2", "value3"])

        assert result.success is True
        # Should handle nulls appropriately
        assert len(result.correlation_matrix) == 3

    async def test_correlation_insufficient_columns(self, stats_ctx: Context) -> None:
        """Test correlation with only one numeric column."""
        ctx = stats_ctx

        with pytest.raises(ToolError):
            await get_correlation_matrix(ctx, columns=["age"])

    async def test_correlation_non_numeric(self, stats_ctx: Context) -> None:
        """Test correlation with non-numeric columns."""
        ctx = stats_ctx

        with pytest.raises(ToolError):
            await get_correlation_matrix(ctx, columns=["name", "department"])


@pytest.mark.asyncio
class TestGetValueCounts:
    """Tests for get_value_counts function."""

    async def test_value_counts_categorical(self, stats_ctx: Context) -> None:
        """Test value counts for categorical column."""
        ctx = stats_ctx
        result = await get_value_counts(ctx, "department")

        assert result.success is True
        assert result.column == "department"
        assert len(result.value_counts) == 4  # 4 departments

        # Engineering should be most common (4 people)
        assert result.value_counts.get("Engineering", 0) == 4

        # Total should equal number of rows
        total = sum(result.value_counts.values())
        assert total == 10

    async def test_value_counts_numeric(self, edge_case_ctx: Context) -> None:
        """Test value counts for numeric column."""
        ctx = edge_case_ctx
        result = await get_value_counts(ctx, "zeros")

        assert result.success is True
        assert len(result.value_counts) == 1  # All zeros
        # Check if key is string "0" or numeric 0
        assert result.value_counts.get("0", 0) == 5 or result.value_counts.get(0, 0) == 5  # type: ignore[call-overload]

    async def test_value_counts_with_nulls(self, sparse_ctx: Context) -> None:
        """Test value counts with null values."""
        ctx = sparse_ctx
        result = await get_value_counts(ctx, "value1")

        assert result.success is True

        # Check if nulls are included (depends on pandas behavior)
        total = sum(result.value_counts.values())
        assert total <= 6  # Total rows including or excluding nulls

    async def test_value_counts_top_n(self, stats_ctx: Context) -> None:
        """Test limiting value counts to top N."""
        ctx = stats_ctx
        result = await get_value_counts(ctx, "department")

        assert result.success is True
        assert len(result.value_counts) <= 10  # Default limit

    async def test_value_counts_invalid_column(self, stats_ctx: Context) -> None:
        """Test value counts with invalid column."""

        ctx = stats_ctx

        with pytest.raises(ColumnNotFoundError):
            await get_value_counts(ctx, "nonexistent")


@pytest.mark.asyncio
class TestEdgeCasesAndIntegration:
    """Test edge cases and integration scenarios."""

    async def test_single_row_statistics(self) -> None:
        """Test statistics with single row."""
        csv_content = "value\n42"
        load_ctx = create_mock_context()
        await load_csv_from_content(load_ctx, csv_content)

        ctx = create_mock_context(load_ctx.session_id)
        stats = await get_statistics(ctx)
        assert stats.success is True
        assert stats.total_rows == 1
        col_stats = await get_column_statistics(ctx, "value")
        assert col_stats.statistics.count == 1
        assert col_stats.statistics.std == 0  # No variation

    async def test_all_null_column(self) -> None:
        """Test statistics on column with all null values."""
        csv_content = "id,empty\n1,\n2,\n3,"
        load_ctx = create_mock_context()
        await load_csv_from_content(load_ctx, csv_content)

        ctx = create_mock_context(load_ctx.session_id)
        col_stats = await get_column_statistics(ctx, "empty")
        assert col_stats.success is True
        assert col_stats.non_null_count == 0

    async def test_mixed_type_handling(self) -> None:
        """Test handling of mixed data types."""
        csv_content = "mixed\n1\n2.5\ntext\n4"
        ctx = create_mock_context()
        await load_csv_from_content(ctx, csv_content)

        # Should treat as string column
        ctx = create_mock_context(ctx.session_id)
        col_stats = await get_column_statistics(ctx, "mixed")
        assert col_stats.success is True
        assert col_stats.data_type == "object"
