"""Unit tests for column server module.

Tests the server wrapper layer, parameter validation, and FastMCP integration for column-level
operations.
"""

import pytest
from fastmcp.exceptions import ToolError

# Ensure full module coverage
import src.databeak.servers.column_server  # noqa: F401
from src.databeak.servers.column_server import (
    add_column,
    change_column_type,
    remove_columns,
    rename_columns,
    select_columns,
    update_column,
)
from src.databeak.servers.io_server import load_csv_from_content
from tests.test_mock_context import create_mock_context, create_mock_context_with_session_data


@pytest.fixture
async def column_session():
    """Create a test session with column operation data."""
    csv_content = """id,first_name,last_name,age,email,salary,is_active,join_date
1,John,Doe,30,john@example.com,50000,true,2023-01-15
2,Jane,Smith,25,jane@test.com,55000,true,2023-02-01
3,Bob,Johnson,35,bob@company.org,60000,false,2023-01-10
4,Alice,Brown,28,alice@example.com,52000,true,2023-03-01"""

    ctx = create_mock_context()
    result = await load_csv_from_content(ctx, csv_content)
    return result.session_id


@pytest.mark.asyncio
class TestColumnServerSelect:
    """Test select_columns server function."""

    @pytest.mark.parametrize(
        "columns,expected_count,description",
        [
            (["first_name", "last_name", "email"], 3, "basic selection"),
            (["email", "id", "first_name"], 3, "reordered selection"),
            (["age"], 1, "single column"),
            (
                [
                    "id",
                    "first_name",
                    "last_name",
                    "age",
                    "email",
                    "salary",
                    "is_active",
                    "join_date",
                ],
                8,
                "all columns",
            ),
        ],
    )
    async def test_select_column_scenarios(
        self, column_session, columns, expected_count, description
    ):
        """Test various column selection scenarios."""
        ctx = create_mock_context_with_session_data(column_session)
        result = await select_columns(ctx, columns)

        assert result.session_id == column_session
        assert result.selected_columns == columns
        assert result.columns_after == expected_count
        if expected_count == 8:  # all columns case
            assert result.columns_before == result.columns_after

    async def test_select_nonexistent_column(self, column_session):
        """Test selecting non-existent column."""
        with pytest.raises(ToolError, match="Column.*not found"):
            ctx = create_mock_context_with_session_data(column_session)
            await select_columns(ctx, ["nonexistent", "first_name"])


@pytest.mark.asyncio
class TestColumnServerRename:
    """Test rename_columns server function."""

    async def test_rename_single_column(self, column_session):
        """Test renaming a single column."""
        mapping = {"first_name": "given_name"}
        ctx = create_mock_context_with_session_data(column_session)
        result = await rename_columns(ctx, mapping)

        assert result.session_id == column_session
        assert result.renamed == mapping
        assert "given_name" in result.columns

    async def test_rename_multiple_columns(self, column_session):
        """Test renaming multiple columns."""
        mapping = {
            "first_name": "given_name",
            "last_name": "family_name",
            "is_active": "active_status",
        }
        ctx = create_mock_context_with_session_data(column_session)
        result = await rename_columns(ctx, mapping)

        assert result.renamed == mapping
        assert len(result.columns) == len(mapping)

    async def test_rename_to_snake_case(self, column_session):
        """Test standardizing column names."""
        mapping = {"join_date": "join_timestamp", "is_active": "active_flag"}
        ctx = create_mock_context_with_session_data(column_session)
        result = await rename_columns(ctx, mapping)

        assert result.renamed == mapping

    async def test_rename_nonexistent_column(self, column_session):
        """Test renaming non-existent column."""
        mapping = {"nonexistent": "new_name"}

        with pytest.raises(ToolError, match="Column.*not found"):
            ctx = create_mock_context_with_session_data(column_session)
            await rename_columns(ctx, mapping)


@pytest.mark.asyncio
class TestColumnServerAdd:
    """Test add_column server function."""

    async def test_add_constant_column(self, column_session):
        """Test adding column with constant value."""
        ctx = create_mock_context_with_session_data(column_session)
        result = await add_column(ctx, "department", "Engineering")

        assert result.session_id == column_session
        assert result.operation == "add"
        assert result.columns_affected == ["department"]
        assert result.rows_affected == 4

    async def test_add_column_with_list(self, column_session):
        """Test adding column with list of values."""
        values = ["Senior", "Junior", "Mid", "Senior"]
        ctx = create_mock_context_with_session_data(column_session)
        result = await add_column(ctx, "level", value=values)

        assert result.operation == "add"
        assert result.columns_affected == ["level"]

    async def test_add_column_with_formula(self, column_session):
        """Test adding computed column with formula."""
        # Use a simpler formula that pandas.eval can handle
        ctx = create_mock_context_with_session_data(column_session)
        result = await add_column(ctx, "age_plus_10", formula="age + 10")

        assert result.operation == "add"
        assert result.columns_affected == ["age_plus_10"]

    async def test_add_column_numeric_formula(self, column_session):
        """Test adding column with numeric calculation."""
        ctx = create_mock_context_with_session_data(column_session)
        result = await add_column(ctx, "monthly_salary", formula="salary / 12")

        assert result.operation == "add"

    async def test_add_duplicate_column_name(self, column_session):
        """Test adding column with existing name."""
        with pytest.raises(ToolError, match="Invalid value"):
            ctx = create_mock_context_with_session_data(column_session)
            await add_column(ctx, "first_name", "test")

    async def test_add_column_invalid_formula(self, column_session):
        """Test adding column with invalid formula."""
        with pytest.raises(ToolError, match="Invalid value"):
            ctx = create_mock_context_with_session_data(column_session)
            await add_column(ctx, "test", formula="invalid_syntax + ")

    async def test_add_column_mismatched_list_length(self, column_session):
        """Test adding column with wrong list length."""
        with pytest.raises(ToolError, match="Invalid value"):
            ctx = create_mock_context_with_session_data(column_session)
            await add_column(ctx, "test", value=[1, 2])  # Only 2 values for 4 rows


@pytest.mark.asyncio
class TestColumnServerRemove:
    """Test remove_columns server function."""

    async def test_remove_single_column(self, column_session):
        """Test removing a single column."""
        ctx = create_mock_context_with_session_data(column_session)
        result = await remove_columns(ctx, ["join_date"])

        assert result.session_id == column_session
        assert result.operation == "remove"
        assert result.columns_affected == ["join_date"]
        assert result.rows_affected == 4

    async def test_remove_multiple_columns(self, column_session):
        """Test removing multiple columns."""
        columns_to_remove = ["salary", "is_active", "join_date"]
        ctx = create_mock_context_with_session_data(column_session)
        result = await remove_columns(ctx, columns_to_remove)

        assert result.columns_affected == columns_to_remove

    async def test_remove_nonexistent_column(self, column_session):
        """Test removing non-existent column."""
        with pytest.raises(ToolError, match="Column.*not found"):
            ctx = create_mock_context_with_session_data(column_session)
            await remove_columns(ctx, ["nonexistent"])


@pytest.mark.asyncio
class TestColumnServerChangeType:
    """Test change_column_type server function."""

    async def test_change_to_int(self, column_session):
        """Test converting column to integer."""
        ctx = create_mock_context_with_session_data(column_session)
        result = await change_column_type(ctx, "age", "int")

        assert result.session_id == column_session
        assert result.operation == "change_type_to_int"
        assert result.columns_affected == ["age"]

    async def test_change_to_float(self, column_session):
        """Test converting column to float."""
        ctx = create_mock_context_with_session_data(column_session)
        result = await change_column_type(ctx, "salary", "float")

        assert result.operation == "change_type_to_float"

    async def test_change_to_string(self, column_session):
        """Test converting column to string."""
        ctx = create_mock_context_with_session_data(column_session)
        result = await change_column_type(ctx, "id", "str")

        assert result.operation == "change_type_to_str"

    async def test_change_to_boolean(self, column_session):
        """Test converting column to boolean."""
        ctx = create_mock_context_with_session_data(column_session)
        result = await change_column_type(ctx, "is_active", "bool")

        assert result.operation == "change_type_to_bool"

    async def test_change_to_datetime(self, column_session):
        """Test converting column to datetime."""
        ctx = create_mock_context_with_session_data(column_session)
        result = await change_column_type(ctx, "join_date", "datetime")

        assert result.operation == "change_type_to_datetime"

    async def test_change_type_with_coerce(self, column_session):
        """Test type conversion with error coercion."""
        ctx = create_mock_context_with_session_data(column_session)
        result = await change_column_type(ctx, "email", "int", errors="coerce")

        assert result.operation == "change_type_to_int"

    async def test_change_type_nonexistent_column(self, column_session):
        """Test changing type of non-existent column."""
        with pytest.raises(ToolError, match="Column.*not found"):
            ctx = create_mock_context_with_session_data(column_session)
            await change_column_type(ctx, "nonexistent", "int")

    async def test_change_type_invalid_type(self, column_session):
        """Test changing to invalid data type."""
        with pytest.raises(ToolError, match="Invalid value"):
            ctx = create_mock_context_with_session_data(column_session)
            await change_column_type(ctx, "age", "invalid_type")


@pytest.mark.asyncio
class TestColumnServerUpdate:
    """Test update_column server function."""

    async def test_update_replace_operation(self, column_session):
        """Test replace operation."""
        ctx = create_mock_context_with_session_data(column_session)
        result = await update_column(
            ctx,
            "first_name",
            {"operation": "replace", "pattern": "John", "replacement": "Jonathan"},
        )

        assert result.operation == "update_replace"
        assert result.columns_affected == ["first_name"]

    async def test_update_map_operation(self, column_session):
        """Test map operation with dictionary."""
        mapping = {"John": "Jonathan", "Jane": "Janet"}
        ctx = create_mock_context_with_session_data(column_session)
        result = await update_column(ctx, "first_name", {"operation": "map", "value": mapping})

        assert result.operation == "update_map"

    async def test_update_fillna_operation(self, column_session):
        """Test fillna operation."""
        ctx = create_mock_context_with_session_data(column_session)
        result = await update_column(ctx, "salary", {"operation": "fillna", "value": 50000})

        assert result.operation == "update_fillna"

    async def test_update_apply_operation(self, column_session):
        """Test apply operation with expression."""
        ctx = create_mock_context_with_session_data(column_session)
        result = await update_column(ctx, "age", {"operation": "apply", "value": "x + 1"})

        assert result.operation == "update_apply"

    async def test_update_replace_missing_params(self, column_session):
        """Test replace operation with missing parameters."""
        with pytest.raises(ToolError, match="Invalid value"):
            ctx = create_mock_context_with_session_data(column_session)
            await update_column(ctx, "first_name", {"operation": "replace", "pattern": "test"})

    async def test_update_map_invalid_value(self, column_session):
        """Test map operation with invalid value type."""
        with pytest.raises(ToolError, match="Invalid value"):
            ctx = create_mock_context_with_session_data(column_session)
            await update_column(ctx, "first_name", {"operation": "map", "value": "not_a_dict"})

    async def test_update_nonexistent_column(self, column_session):
        """Test updating non-existent column."""
        with pytest.raises(ToolError, match="Column.*not found"):
            ctx = create_mock_context_with_session_data(column_session)
            await update_column(ctx, "nonexistent", {"operation": "fillna", "value": 0})


@pytest.mark.asyncio
class TestColumnServerErrorHandling:
    """Test error handling in column server."""

    async def test_operations_invalid_session(self):
        """Test operations with invalid session ID."""
        invalid_session = "invalid-session-id"
        ctx = create_mock_context(invalid_session)

        with pytest.raises(ToolError, match="No data loaded in session"):
            await select_columns(ctx, ["test"])

        with pytest.raises(ToolError, match="No data loaded in session"):
            await rename_columns(ctx, {"old": "new"})

        with pytest.raises(ToolError, match="No data loaded in session"):
            await add_column(ctx, "test", "value")

        with pytest.raises(ToolError, match="No data loaded in session"):
            await remove_columns(ctx, ["test"])

        with pytest.raises(ToolError, match="No data loaded in session"):
            await change_column_type(ctx, "test", "int")

        with pytest.raises(ToolError, match="No data loaded in session"):
            await update_column(ctx, "test", {"operation": "fillna", "value": 0})
