"""Comprehensive tests for column_server update_column function with discriminated unions."""

import pytest
from fastmcp.exceptions import ToolError

from src.databeak.servers.column_server import (
    ApplyOperation,
    FillNaOperation,
    MapOperation,
    ReplaceOperation,
    update_column,
)
from src.databeak.servers.io_server import load_csv_from_content
from tests.test_mock_context import create_mock_context, create_mock_context_with_session_data


@pytest.fixture
async def update_session():
    """Create a test session with data for update operations."""
    csv_content = """id,name,category,value,status
1,Item A,Category 1,100,active
2,Item B,Category 2,200,inactive
3,Item C,Category 1,,active
4,Item D,Category 2,400,
5,Item E,Category 3,500,inactive"""

    ctx = create_mock_context()
    result = await load_csv_from_content(ctx, csv_content)
    return ctx.session_id


class TestUpdateColumnDiscriminatedUnions:
    """Test update_column with discriminated union operations."""

    async def test_replace_operation_object(self, update_session):
        """Test update_column with ReplaceOperation object."""
        operation = ReplaceOperation(pattern="Category 1", replacement="Cat-1")

        result = await update_column(
            create_mock_context_with_session_data(update_session),
            column="category",
            operation=operation,
        )

        assert result.success is True
        assert "category" in result.columns_affected
        assert result.operation == "update_replace"
        assert result.rows_affected > 0

    async def test_map_operation_object(self, update_session):
        """Test update_column with MapOperation object."""
        operation = MapOperation(mapping={"active": 1, "inactive": 0})

        result = await update_column(
            create_mock_context_with_session_data(update_session),
            column="status",
            operation=operation,
        )

        assert result.success is True
        assert "status" in result.columns_affected
        assert result.operation == "update_map"

    async def test_apply_operation_object(self, update_session):
        """Test update_column with ApplyOperation object."""
        operation = ApplyOperation(expression="x * 2")

        result = await update_column(
            create_mock_context_with_session_data(update_session),
            column="value",
            operation=operation,
        )

        assert result.success is True
        assert "value" in result.columns_affected
        assert result.operation == "update_apply"

    async def test_fillna_operation_object(self, update_session):
        """Test update_column with FillNaOperation object."""
        operation = FillNaOperation(value=0)

        result = await update_column(
            create_mock_context_with_session_data(update_session),
            column="value",
            operation=operation,
        )

        assert result.success is True
        assert "value" in result.columns_affected
        assert result.operation == "update_fillna"

    async def test_replace_operation_dict(self, update_session):
        """Test update_column with replace operation as dict."""
        operation = {"type": "replace", "pattern": "Item A", "replacement": "Product A"}

        result = await update_column(
            create_mock_context_with_session_data(update_session),
            column="name",
            operation=operation,
        )

        assert result.success is True
        assert result.operation == "update_replace"

    async def test_map_operation_dict(self, update_session):
        """Test update_column with map operation as dict."""
        operation = {
            "type": "map",
            "mapping": {"Category 1": "Group A", "Category 2": "Group B", "Category 3": "Group C"},
        }

        result = await update_column(
            create_mock_context_with_session_data(update_session),
            column="category",
            operation=operation,
        )

        assert result.success is True
        assert result.operation == "update_map"

    async def test_apply_operation_dict(self, update_session):
        """Test update_column with apply operation as dict."""
        operation = {"type": "apply", "expression": "x.upper()"}

        result = await update_column(
            create_mock_context_with_session_data(update_session),
            column="status",
            operation=operation,
        )

        assert result.success is True
        assert result.operation == "update_apply"

    async def test_fillna_operation_dict(self, update_session):
        """Test update_column with fillna operation as dict."""
        operation = {"type": "fillna", "value": "unknown"}

        result = await update_column(
            create_mock_context_with_session_data(update_session),
            column="status",
            operation=operation,
        )

        assert result.success is True
        assert result.operation == "update_fillna"

    async def test_invalid_expression_apply(self, update_session):
        """Test apply operation with invalid expression."""
        operation = ApplyOperation(
            expression="import os; os.system('ls')"  # Dangerous expression
        )

        with pytest.raises(ToolError, match="Invalid value for parameter"):
            await update_column(
                create_mock_context_with_session_data(update_session),
                column="value",
                operation=operation,
            )

    async def test_invalid_operation_type_dict(self, update_session):
        """Test with invalid operation type in dict."""
        operation = {"type": "invalid_op", "value": 123}

        with pytest.raises(ToolError, match="Invalid value for parameter"):
            await update_column(
                create_mock_context_with_session_data(update_session),
                column="value",
                operation=operation,
            )

    async def test_legacy_fillna_format(self, update_session):
        """Test legacy fillna format (backward compatibility)."""
        # Legacy format without type field
        operation = {"operation": "fillna", "value": -1}

        result = await update_column(
            create_mock_context_with_session_data(update_session),
            column="value",
            operation=operation,
        )

        assert result.success is True
        assert result.operation == "update_fillna"

    async def test_column_not_found(self, update_session):
        """Test update_column with non-existent column."""
        operation = FillNaOperation(value=0)

        with pytest.raises(ToolError, match="Column 'nonexistent' not found"):
            await update_column(
                create_mock_context_with_session_data(update_session),
                column="nonexistent",
                operation=operation,
            )

    async def test_mixed_operations_sequence(self, update_session):
        """Test sequence of different operations."""
        # First, fill nulls
        result1 = await update_column(
            create_mock_context_with_session_data(update_session),
            column="value",
            operation=FillNaOperation(value=0),
        )
        assert result1.success is True

        # Then apply transformation
        result2 = await update_column(
            create_mock_context_with_session_data(update_session),
            column="value",
            operation=ApplyOperation(expression="x * 1.1"),
        )
        assert result2.success is True

        # Finally, map categories
        result3 = await update_column(
            create_mock_context_with_session_data(update_session),
            column="category",
            operation=MapOperation(
                mapping={"Category 1": "Premium", "Category 2": "Standard", "Category 3": "Basic"}
            ),
        )
        assert result3.success is True
