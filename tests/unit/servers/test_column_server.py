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


@pytest.fixture
async def column_session():
    """Create a test session with column operation data."""
    csv_content = """id,first_name,last_name,age,email,salary,is_active,join_date
1,John,Doe,30,john@example.com,50000,true,2023-01-15
2,Jane,Smith,25,jane@test.com,55000,true,2023-02-01
3,Bob,Johnson,35,bob@company.org,60000,false,2023-01-10
4,Alice,Brown,28,alice@example.com,52000,true,2023-03-01"""

    result = await load_csv_from_content(csv_content)
    return result.session_id


@pytest.mark.asyncio
class TestColumnServerSelect:
    """Test select_columns server function."""

    async def test_select_basic_columns(self, column_session):
        """Test selecting basic set of columns."""
        result = await select_columns(column_session, ["first_name", "last_name", "email"])

        assert result.session_id == column_session
        assert result.selected_columns == ["first_name", "last_name", "email"]
        assert result.columns_before == 8
        assert result.columns_after == 3

    async def test_select_reorder_columns(self, column_session):
        """Test column reordering through selection."""
        result = await select_columns(column_session, ["email", "id", "first_name"])

        assert result.selected_columns == ["email", "id", "first_name"]
        assert result.columns_after == 3

    async def test_select_single_column(self, column_session):
        """Test selecting single column."""
        result = await select_columns(column_session, ["age"])

        assert result.columns_after == 1
        assert result.selected_columns == ["age"]

    async def test_select_all_columns(self, column_session):
        """Test selecting all existing columns."""
        all_cols = [
            "id",
            "first_name",
            "last_name",
            "age",
            "email",
            "salary",
            "is_active",
            "join_date",
        ]
        result = await select_columns(column_session, all_cols)

        assert result.columns_before == result.columns_after

    async def test_select_nonexistent_column(self, column_session):
        """Test selecting non-existent column."""
        with pytest.raises(ToolError, match="not found"):
            await select_columns(column_session, ["nonexistent", "first_name"])


@pytest.mark.asyncio
class TestColumnServerRename:
    """Test rename_columns server function."""

    async def test_rename_single_column(self, column_session):
        """Test renaming a single column."""
        mapping = {"first_name": "given_name"}
        result = await rename_columns(column_session, mapping)

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
        result = await rename_columns(column_session, mapping)

        assert result.renamed == mapping
        assert len(result.columns) == len(mapping)

    async def test_rename_to_snake_case(self, column_session):
        """Test standardizing column names."""
        mapping = {"join_date": "join_timestamp", "is_active": "active_flag"}
        result = await rename_columns(column_session, mapping)

        assert result.renamed == mapping

    async def test_rename_nonexistent_column(self, column_session):
        """Test renaming non-existent column."""
        mapping = {"nonexistent": "new_name"}

        with pytest.raises(ToolError, match="not found"):
            await rename_columns(column_session, mapping)


@pytest.mark.asyncio
class TestColumnServerAdd:
    """Test add_column server function."""

    async def test_add_constant_column(self, column_session):
        """Test adding column with constant value."""
        result = await add_column(column_session, "department", "Engineering")

        assert result.session_id == column_session
        assert result.operation == "add"
        assert result.columns_affected == ["department"]
        assert result.rows_affected == 4

    async def test_add_column_with_list(self, column_session):
        """Test adding column with list of values."""
        values = ["Senior", "Junior", "Mid", "Senior"]
        result = await add_column(column_session, "level", value=values)

        assert result.operation == "add"
        assert result.columns_affected == ["level"]

    async def test_add_column_with_formula(self, column_session):
        """Test adding computed column with formula."""
        # Use a simpler formula that pandas.eval can handle
        result = await add_column(column_session, "age_plus_10", formula="age + 10")

        assert result.operation == "add"
        assert result.columns_affected == ["age_plus_10"]

    async def test_add_column_numeric_formula(self, column_session):
        """Test adding column with numeric calculation."""
        result = await add_column(column_session, "monthly_salary", formula="salary / 12")

        assert result.operation == "add"

    async def test_add_duplicate_column_name(self, column_session):
        """Test adding column with existing name."""
        with pytest.raises(ToolError, match="Invalid value"):
            await add_column(column_session, "first_name", "test")

    async def test_add_column_invalid_formula(self, column_session):
        """Test adding column with invalid formula."""
        with pytest.raises(ToolError, match="Invalid value"):
            await add_column(column_session, "test", formula="invalid_syntax + ")

    async def test_add_column_mismatched_list_length(self, column_session):
        """Test adding column with wrong list length."""
        with pytest.raises(ToolError, match="Invalid value"):
            await add_column(column_session, "test", value=[1, 2])  # Only 2 values for 4 rows


@pytest.mark.asyncio
class TestColumnServerRemove:
    """Test remove_columns server function."""

    async def test_remove_single_column(self, column_session):
        """Test removing a single column."""
        result = await remove_columns(column_session, ["join_date"])

        assert result.session_id == column_session
        assert result.operation == "remove"
        assert result.columns_affected == ["join_date"]
        assert result.rows_affected == 4

    async def test_remove_multiple_columns(self, column_session):
        """Test removing multiple columns."""
        columns_to_remove = ["salary", "is_active", "join_date"]
        result = await remove_columns(column_session, columns_to_remove)

        assert result.columns_affected == columns_to_remove

    async def test_remove_nonexistent_column(self, column_session):
        """Test removing non-existent column."""
        with pytest.raises(ToolError, match="not found"):
            await remove_columns(column_session, ["nonexistent"])


@pytest.mark.asyncio
class TestColumnServerChangeType:
    """Test change_column_type server function."""

    async def test_change_to_int(self, column_session):
        """Test converting column to integer."""
        result = await change_column_type(column_session, "age", "int")

        assert result.session_id == column_session
        assert result.operation == "change_type_to_int"
        assert result.columns_affected == ["age"]

    async def test_change_to_float(self, column_session):
        """Test converting column to float."""
        result = await change_column_type(column_session, "salary", "float")

        assert result.operation == "change_type_to_float"

    async def test_change_to_string(self, column_session):
        """Test converting column to string."""
        result = await change_column_type(column_session, "id", "str")

        assert result.operation == "change_type_to_str"

    async def test_change_to_boolean(self, column_session):
        """Test converting column to boolean."""
        result = await change_column_type(column_session, "is_active", "bool")

        assert result.operation == "change_type_to_bool"

    async def test_change_to_datetime(self, column_session):
        """Test converting column to datetime."""
        result = await change_column_type(column_session, "join_date", "datetime")

        assert result.operation == "change_type_to_datetime"

    async def test_change_type_with_coerce(self, column_session):
        """Test type conversion with error coercion."""
        result = await change_column_type(column_session, "email", "int", errors="coerce")

        assert result.operation == "change_type_to_int"

    async def test_change_type_nonexistent_column(self, column_session):
        """Test changing type of non-existent column."""
        with pytest.raises(ToolError, match="not found"):
            await change_column_type(column_session, "nonexistent", "int")

    async def test_change_type_invalid_type(self, column_session):
        """Test changing to invalid data type."""
        with pytest.raises(ToolError, match="Invalid value"):
            await change_column_type(column_session, "age", "invalid_type")


@pytest.mark.asyncio
class TestColumnServerUpdate:
    """Test update_column server function."""

    async def test_update_replace_operation(self, column_session):
        """Test replace operation."""
        result = await update_column(
            column_session,
            "first_name",
            {"operation": "replace", "pattern": "John", "replacement": "Jonathan"},
        )

        assert result.operation == "update_replace"
        assert result.columns_affected == ["first_name"]

    async def test_update_map_operation(self, column_session):
        """Test map operation with dictionary."""
        mapping = {"John": "Jonathan", "Jane": "Janet"}
        result = await update_column(
            column_session, "first_name", {"operation": "map", "value": mapping}
        )

        assert result.operation == "update_map"

    async def test_update_fillna_operation(self, column_session):
        """Test fillna operation."""
        result = await update_column(
            column_session, "salary", {"operation": "fillna", "value": 50000}
        )

        assert result.operation == "update_fillna"

    async def test_update_apply_operation(self, column_session):
        """Test apply operation with expression."""
        result = await update_column(
            column_session, "age", {"operation": "apply", "value": "x + 1"}
        )

        assert result.operation == "update_apply"

    async def test_update_replace_missing_params(self, column_session):
        """Test replace operation with missing parameters."""
        with pytest.raises(ToolError, match="Invalid value"):
            await update_column(
                column_session, "first_name", {"operation": "replace", "pattern": "test"}
            )

    async def test_update_map_invalid_value(self, column_session):
        """Test map operation with invalid value type."""
        with pytest.raises(ToolError, match="Invalid value"):
            await update_column(
                column_session, "first_name", {"operation": "map", "value": "not_a_dict"}
            )

    async def test_update_nonexistent_column(self, column_session):
        """Test updating non-existent column."""
        with pytest.raises(ToolError, match="not found"):
            await update_column(column_session, "nonexistent", {"operation": "fillna", "value": 0})


@pytest.mark.asyncio
class TestColumnServerErrorHandling:
    """Test error handling in column server."""

    async def test_operations_invalid_session(self):
        """Test operations with invalid session ID."""
        invalid_session = "invalid-session-id"

        with pytest.raises(ToolError, match="not found"):
            await select_columns(invalid_session, ["test"])

        with pytest.raises(ToolError, match="not found"):
            await rename_columns(invalid_session, {"old": "new"})

        with pytest.raises(ToolError, match="not found"):
            await add_column(invalid_session, "test", "value")

        with pytest.raises(ToolError, match="not found"):
            await remove_columns(invalid_session, ["test"])

        with pytest.raises(ToolError, match="not found"):
            await change_column_type(invalid_session, "test", "int")

        with pytest.raises(ToolError, match="not found"):
            await update_column(invalid_session, "test", {"operation": "fillna", "value": 0})
