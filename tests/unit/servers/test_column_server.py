"""Unit Tests the server wrapper layer, parameter validation, and FastMCP integration for column-
level operations."""

from typing import cast

import pytest
from fastmcp import Context
from fastmcp.exceptions import ToolError

# Ensure full module coverage
import databeak.servers.column_server  # noqa: F401
from databeak.core.session import get_session_manager
from databeak.servers.column_server import (
    add_column,
    change_column_type,
    remove_columns,
    rename_columns,
    select_columns,
    update_column,
)
from databeak.servers.io_server import load_csv_from_content
from tests.test_mock_context import create_mock_context


@pytest.fixture
async def ctx_fixture() -> Context:
    """Create a test session with column operation data."""
    csv_content = """id,first_name,last_name,age,email,salary,is_active,join_date
1,John,Doe,30,john@example.com,50000,true,2023-01-15
2,Jane,Smith,25,jane@test.com,55000,true,2023-02-01
3,Bob,Johnson,35,bob@company.org,60000,false,2023-01-10
4,Alice,Brown,28,alice@example.com,52000,true,2023-03-01"""

    ctx = create_mock_context()
    _session = get_session_manager().get_or_create_session(ctx.session_id)
    _result = await load_csv_from_content(ctx, csv_content)
    return ctx


@pytest.mark.asyncio
class TestColumnServerSelect:
    """Test select_columns server function."""

    @pytest.mark.parametrize(
        ("columns", "expected_count", "description"),
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
        self, ctx_fixture: Context, columns: list[str], expected_count: int, description: str
    ) -> None:
        """Test various column selection scenarios."""
        result = await select_columns(ctx_fixture, columns)

        assert result.selected_columns == columns
        assert result.columns_after == expected_count
        if expected_count == 8:  # all columns case
            assert result.columns_before == result.columns_after

    async def test_select_nonexistent_column(self, ctx_fixture: Context) -> None:
        """Test selecting non-existent column."""
        with pytest.raises(ToolError, match="Column.*not found"):
            await select_columns(ctx_fixture, ["nonexistent", "first_name"])


@pytest.mark.asyncio
class TestColumnServerRename:
    """Test rename_columns server function."""

    async def test_rename_single_column(self, ctx_fixture: Context) -> None:
        """Test renaming a single column."""
        mapping = {"first_name": "given_name"}
        result = await rename_columns(ctx_fixture, mapping)

        assert result.renamed == mapping
        assert "given_name" in result.columns

    async def test_rename_multiple_columns(self, ctx_fixture: Context) -> None:
        """Test renaming multiple columns."""
        mapping = {
            "first_name": "given_name",
            "last_name": "family_name",
            "is_active": "active_status",
        }
        result = await rename_columns(ctx_fixture, mapping)

        assert result.renamed == mapping
        assert len(result.columns) == len(mapping)

    async def test_rename_to_snake_case(self, ctx_fixture: Context) -> None:
        """Test standardizing column names."""
        mapping = {"join_date": "join_timestamp", "is_active": "active_flag"}
        result = await rename_columns(ctx_fixture, mapping)

        assert result.renamed == mapping

    async def test_rename_nonexistent_column(self, ctx_fixture: Context) -> None:
        """Test renaming non-existent column."""
        mapping = {"nonexistent": "new_name"}

        with pytest.raises(ToolError, match="Column.*not found"):
            await rename_columns(ctx_fixture, mapping)


@pytest.mark.asyncio
class TestColumnServerAdd:
    """Test add_column server function."""

    async def test_add_constant_column(self, ctx_fixture: Context) -> None:
        """Test adding column with constant value."""
        result = await add_column(ctx_fixture, "department", "Engineering")

        assert result.operation == "add"
        assert result.columns_affected == ["department"]
        assert result.rows_affected == 4

    async def test_add_column_with_list(self, ctx_fixture: Context) -> None:
        """Test adding column with list of values."""
        values = ["Senior", "Junior", "Mid", "Senior"]
        result = await add_column(
            ctx_fixture, "level", value=cast(list[str | int | float | bool | None], values)
        )

        assert result.operation == "add"
        assert result.columns_affected == ["level"]

    async def test_add_column_with_formula(self, ctx_fixture: Context) -> None:
        """Test adding computed column with formula."""
        # Use a simpler formula that pandas.eval can handle
        result = await add_column(ctx_fixture, "age_plus_10", formula="age + 10")

        assert result.operation == "add"
        assert result.columns_affected == ["age_plus_10"]

    async def test_add_column_numeric_formula(self, ctx_fixture: Context) -> None:
        """Test adding column with numeric calculation."""
        result = await add_column(ctx_fixture, "monthly_salary", formula="salary / 12")

        assert result.operation == "add"

    async def test_add_duplicate_column_name(self, ctx_fixture: Context) -> None:
        """Test adding column with existing name."""
        with pytest.raises(ToolError, match="Invalid value"):
            await add_column(ctx_fixture, "first_name", "test")

    async def test_add_column_invalid_formula(self, ctx_fixture: Context) -> None:
        """Test adding column with invalid formula."""
        with pytest.raises(ToolError, match="Invalid value"):
            await add_column(ctx_fixture, "test", formula="invalid_syntax + ")

    async def test_add_column_mismatched_list_length(self, ctx_fixture: Context) -> None:
        """Test adding column with wrong list length."""
        with pytest.raises(ToolError, match="Invalid value"):
            await add_column(ctx_fixture, "test", value=[1, 2])  # Only 2 values for 4 rows


@pytest.mark.asyncio
class TestColumnServerRemove:
    """Test remove_columns server function."""

    async def test_remove_single_column(self, ctx_fixture: Context) -> None:
        """Test removing a single column."""
        result = await remove_columns(ctx_fixture, ["join_date"])

        assert result.operation == "remove"
        assert result.columns_affected == ["join_date"]
        assert result.rows_affected == 4

    async def test_remove_multiple_columns(self, ctx_fixture: Context) -> None:
        """Test removing multiple columns."""
        columns_to_remove = ["salary", "is_active", "join_date"]
        result = await remove_columns(ctx_fixture, columns_to_remove)

        assert result.columns_affected == columns_to_remove

    async def test_remove_nonexistent_column(self, ctx_fixture: Context) -> None:
        """Test removing non-existent column."""
        with pytest.raises(ToolError, match="Column.*not found"):
            await remove_columns(ctx_fixture, ["nonexistent"])


@pytest.mark.asyncio
class TestColumnServerChangeType:
    """Test change_column_type server function."""

    async def test_change_to_int(self, ctx_fixture: Context) -> None:
        """Test converting column to integer."""
        result = await change_column_type(ctx_fixture, "age", "int")

        assert result.operation == "change_type_to_int"
        assert result.columns_affected == ["age"]

    async def test_change_to_float(self, ctx_fixture: Context) -> None:
        """Test converting column to float."""
        result = await change_column_type(ctx_fixture, "salary", "float")

        assert result.operation == "change_type_to_float"

    async def test_change_to_string(self, ctx_fixture: Context) -> None:
        """Test converting column to string."""
        result = await change_column_type(ctx_fixture, "id", "str")

        assert result.operation == "change_type_to_str"

    async def test_change_to_boolean(self, ctx_fixture: Context) -> None:
        """Test converting column to boolean."""
        result = await change_column_type(ctx_fixture, "is_active", "bool")

        assert result.operation == "change_type_to_bool"

    async def test_change_to_datetime(self, ctx_fixture: Context) -> None:
        """Test converting column to datetime."""
        result = await change_column_type(ctx_fixture, "join_date", "datetime")

        assert result.operation == "change_type_to_datetime"

    async def test_change_type_with_coerce(self, ctx_fixture: Context) -> None:
        """Test type conversion with error coercion."""
        result = await change_column_type(ctx_fixture, "email", "int", errors="coerce")

        assert result.operation == "change_type_to_int"

    async def test_change_type_nonexistent_column(self, ctx_fixture: Context) -> None:
        """Test changing type of non-existent column."""
        with pytest.raises(ToolError, match="Column.*not found"):
            await change_column_type(ctx_fixture, "nonexistent", "int")

    async def test_change_type_invalid_type(self, ctx_fixture: Context) -> None:
        """Test changing to invalid data type."""
        with pytest.raises(ToolError, match="Invalid value"):
            await change_column_type(ctx_fixture, "age", "invalid_type")  # type: ignore[arg-type]


@pytest.mark.asyncio
class TestColumnServerUpdate:
    """Test update_column server function."""

    async def test_update_replace_operation(self, ctx_fixture: Context) -> None:
        """Test replace operation."""
        result = await update_column(
            ctx_fixture,
            "first_name",
            {"operation": "replace", "pattern": "John", "replacement": "Jonathan"},
        )

        assert result.operation == "update_replace"
        assert result.columns_affected == ["first_name"]

    async def test_update_map_operation(self, ctx_fixture: Context) -> None:
        """Test map operation with dictionary."""
        mapping = {"John": "Jonathan", "Jane": "Janet"}
        result = await update_column(
            ctx_fixture,
            "first_name",
            {"operation": "map", "value": mapping},
        )

        assert result.operation == "update_map"

    async def test_update_fillna_operation(self, ctx_fixture: Context) -> None:
        """Test fillna operation."""
        result = await update_column(ctx_fixture, "salary", {"operation": "fillna", "value": 50000})

        assert result.operation == "update_fillna"

    async def test_update_apply_operation(self, ctx_fixture: Context) -> None:
        """Test apply operation with expression."""
        result = await update_column(ctx_fixture, "age", {"operation": "apply", "value": "x + 1"})

        assert result.operation == "update_apply"

    async def test_update_replace_missing_params(self, ctx_fixture: Context) -> None:
        """Test replace operation with missing parameters."""
        with pytest.raises(ToolError, match="Invalid value"):
            await update_column(
                ctx_fixture,
                "first_name",
                {"operation": "replace", "pattern": "test"},
            )

    async def test_update_map_invalid_value(self, ctx_fixture: Context) -> None:
        """Test map operation with invalid value type."""
        with pytest.raises(ToolError, match="Invalid value"):
            await update_column(
                ctx_fixture,
                "first_name",
                {"operation": "map", "value": "not_a_dict"},
            )

    async def test_update_nonexistent_column(self, ctx_fixture: Context) -> None:
        """Test updating non-existent column."""
        with pytest.raises(ToolError, match="Column.*not found"):
            await update_column(ctx_fixture, "nonexistent", {"operation": "fillna", "value": 0})


@pytest.mark.asyncio
class TestColumnServerErrorHandling:
    """Test error handling in column server."""

    async def test_operations_invalid_session(self) -> None:
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
