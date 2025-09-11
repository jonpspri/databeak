"""Unit tests for transformation server module.

Tests the server wrapper layer, Pydantic model conversion, and FastMCP integration for core data
transformation operations.
"""

import pytest
from fastmcp.exceptions import ToolError

# Ensure full module coverage
import src.databeak.servers.transformation_server  # noqa: F401
from src.databeak.servers.io_server import load_csv_from_content
from src.databeak.servers.transformation_server import (
    FilterCondition,
    SortColumn,
    fill_missing_values,
    filter_rows,
    remove_duplicates,
    sort_data,
)


@pytest.fixture
async def transformation_session():
    """Create a test session with transformation data."""
    csv_content = """name,age,score,status,notes
John Doe,30,85.5,active,Good performance
Jane Smith,25,92.0,active,
Bob Johnson,35,78.5,inactive,Needs improvement
Alice Brown,28,95.0,active,Excellent
Charlie Wilson,30,85.5,pending,"""

    result = await load_csv_from_content(csv_content)
    return result.session_id


@pytest.mark.asyncio
class TestTransformationServerFilterRows:
    """Test filter_rows server function."""

    async def test_filter_with_pydantic_models(self, transformation_session):
        """Test filtering using Pydantic FilterCondition models."""
        conditions = [
            FilterCondition(column="age", operator=">", value=27),
            FilterCondition(column="status", operator="==", value="active"),
        ]

        result = filter_rows(transformation_session, conditions, mode="and")
        assert result.success is True
        assert result.rows_before == 5
        assert result.rows_after == 2  # John (30, active) and Alice (28 > 27, active)

    async def test_filter_with_mixed_formats(self, transformation_session):
        """Test filtering with mixed Pydantic models and dict formats."""
        conditions = [
            FilterCondition(column="age", operator=">=", value=30),
            {"column": "status", "operator": "!=", "value": "inactive"},
        ]

        result = filter_rows(transformation_session, conditions, mode="and")
        assert result.success is True
        assert result.rows_after == 2  # John and Charlie

    async def test_filter_null_operators(self, transformation_session):
        """Test null operators with Pydantic models."""
        conditions = [FilterCondition(column="notes", operator="is_null")]

        result = filter_rows(transformation_session, conditions)
        assert result.success is True
        assert result.rows_after == 2  # Jane and Charlie have empty notes

    async def test_filter_text_operators(self, transformation_session):
        """Test text operators with Pydantic models."""
        conditions = [FilterCondition(column="name", operator="contains", value="o")]

        result = filter_rows(transformation_session, conditions)
        assert result.success is True
        assert result.rows_after == 4  # John Doe, Bob Johnson, Alice Brown, Charlie Wilson

    async def test_filter_or_mode(self, transformation_session):
        """Test OR mode with multiple conditions."""
        conditions = [
            FilterCondition(column="age", operator="<", value=28),
            FilterCondition(column="score", operator=">", value=94),
        ]

        result = filter_rows(transformation_session, conditions, mode="or")
        assert result.success is True
        assert result.rows_after == 2  # Jane (age < 28) and Alice (score > 94)


@pytest.mark.asyncio
class TestTransformationServerSort:
    """Test sort_data server function."""

    async def test_sort_with_pydantic_models(self, transformation_session):
        """Test sorting using Pydantic SortColumn models."""
        columns = [
            SortColumn(column="age", ascending=False),
            SortColumn(column="score", ascending=True),
        ]

        result = sort_data(transformation_session, columns)
        assert result.sorted_by == ["age", "score"]
        assert len(result.ascending) == 2

    async def test_sort_with_mixed_formats(self, transformation_session):
        """Test sorting with mixed formats."""
        columns = [
            SortColumn(column="status", ascending=True),
            {"column": "score", "ascending": False},
            "name",  # Simple string format
        ]

        result = sort_data(transformation_session, columns)
        assert result.success is True

    async def test_sort_string_columns(self, transformation_session):
        """Test sorting simple string columns."""
        result = sort_data(transformation_session, ["name", "status"])
        assert result.success is True


@pytest.mark.asyncio
class TestTransformationServerDuplicates:
    """Test remove_duplicates server function."""

    async def test_remove_duplicates_all_columns(self, transformation_session):
        """Test removing exact duplicates."""
        # First add a duplicate row
        from src.databeak.tools.transformations import insert_row

        await insert_row(
            transformation_session,
            -1,
            {
                "name": "John Doe",
                "age": 30,
                "score": 85.5,
                "status": "active",
                "notes": "Good performance",
            },
        )

        result = remove_duplicates(transformation_session)
        assert result.operation == "remove_duplicates"
        assert result.rows_affected > 0

    async def test_remove_duplicates_subset(self, transformation_session):
        """Test removing duplicates based on subset of columns."""
        result = remove_duplicates(transformation_session, subset=["age", "score"])
        assert result.success is True

    async def test_remove_duplicates_keep_options(self, transformation_session):
        """Test different keep options."""
        for keep_option in ["first", "last", "none"]:
            result = remove_duplicates(transformation_session, keep=keep_option)
            assert result.operation == "remove_duplicates"


@pytest.mark.asyncio
class TestTransformationServerFillMissing:
    """Test fill_missing_values server function."""

    async def test_fill_missing_drop(self, transformation_session):
        """Test dropping rows with missing values."""
        result = fill_missing_values(transformation_session, strategy="drop")
        assert result.operation == "fill_missing_values"
        assert result.success is True

    async def test_fill_missing_with_value(self, transformation_session):
        """Test filling with specific value."""
        result = fill_missing_values(transformation_session, strategy="fill", value="Unknown")
        assert result.success is True

    async def test_fill_missing_forward_fill(self, transformation_session):
        """Test forward fill strategy."""
        result = fill_missing_values(transformation_session, strategy="forward")
        assert result.success is True

    async def test_fill_missing_column_specific(self, transformation_session):
        """Test filling specific columns only."""
        result = fill_missing_values(
            transformation_session, strategy="fill", value="N/A", columns=["notes"]
        )
        assert result.success is True

    async def test_fill_missing_statistical(self, transformation_session):
        """Test statistical fill strategies."""
        for strategy in ["mean", "median", "mode"]:
            result = fill_missing_values(transformation_session, strategy=strategy)
            assert result.success is True


@pytest.mark.asyncio
class TestTransformationServerErrorHandling:
    """Test error handling in transformation server."""

    async def test_filter_invalid_session(self):
        """Test filtering with invalid session."""
        conditions = [FilterCondition(column="test", operator="==", value="test")]

        with pytest.raises(ToolError, match="Invalid session"):
            filter_rows("invalid-session-id", conditions)

    async def test_sort_invalid_session(self):
        """Test sorting with invalid session."""
        columns = [SortColumn(column="test", ascending=True)]

        with pytest.raises(ToolError, match="Invalid session"):
            sort_data("invalid-session-id", columns)

    async def test_filter_invalid_column(self, transformation_session):
        """Test filtering with invalid column name."""
        conditions = [FilterCondition(column="nonexistent", operator="==", value="test")]

        with pytest.raises(ToolError, match="not found"):
            filter_rows(transformation_session, conditions)

    async def test_sort_invalid_column(self, transformation_session):
        """Test sorting with invalid column name."""
        columns = [SortColumn(column="nonexistent", ascending=True)]

        with pytest.raises(ToolError):
            sort_data(transformation_session, columns)
