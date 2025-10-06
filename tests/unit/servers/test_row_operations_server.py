"""Unit tests for row operations server module.

Tests the server wrapper layer, parameter validation, and FastMCP integration for row-level
operations with comprehensive boundary checking and error handling.
"""

import json

import pytest
from fastmcp import Context
from fastmcp.exceptions import ToolError

# Ensure full module coverage
import databeak.servers.row_operations_server  # noqa: F401
from databeak.exceptions import (
    ColumnNotFoundError,
    InvalidParameterError,
    NoDataLoadedError,
    SessionNotFoundError,
)
from databeak.models import CellValue
from databeak.servers.io_server import load_csv_from_content
from databeak.servers.row_operations_server import (
    delete_row,
    get_cell_value,
    get_column_data,
    get_row_data,
    insert_row,
    set_cell_value,
    update_row,
)
from tests.test_mock_context import create_mock_context


@pytest.fixture
async def row_operations_ctx() -> Context:
    """Create a test session with row operations test data."""
    csv_content = """id,first_name,last_name,age,email,salary,is_active,join_date
1,John,Doe,30,john@example.com,50000,true,2023-01-15
2,Jane,Smith,25,jane@test.com,55000,true,2023-02-01
3,Bob,Johnson,35,bob@company.org,60000,false,2023-01-10
4,Alice,Brown,28,alice@example.com,52000,true,2023-03-01"""

    ctx = create_mock_context()
    await load_csv_from_content(ctx, csv_content)
    return ctx


@pytest.mark.asyncio
class TestGetCellValue:
    """Test get_cell_value server function."""

    @pytest.mark.parametrize(
        ("row_index", "column", "expected_value", "description"),
        [
            (0, "first_name", "John", "string value by column name"),
            (1, "age", 25, "integer value by column name"),
            (2, "is_active", False, "boolean value by column name"),
            (0, 0, 1, "value by column index"),
            (1, 1, "Jane", "string value by column index"),
            (3, 5, 52000, "numeric value by column index"),
        ],
    )
    async def test_get_cell_value_success(
        self,
        row_operations_ctx: Context,
        row_index: int,
        column: int | str,
        expected_value: CellValue,
        description: str,
    ) -> None:
        """Test successful cell value retrieval with various coordinates."""
        ctx = row_operations_ctx
        result = get_cell_value(ctx, row_index, column)

        assert result.success is True
        assert result.value == expected_value
        assert result.coordinates["row"] == row_index
        assert isinstance(result.coordinates["column"], str)  # Always returns column name
        assert result.data_type is not None

    async def test_get_cell_value_null_handling(self, row_operations_ctx: Context) -> None:
        """Test null value handling in cell retrieval."""
        # First set a cell to None to test null handling
        ctx = row_operations_ctx
        set_cell_value(ctx, 0, "email", None)

        result = get_cell_value(ctx, 0, "email")

        assert result.success is True
        assert result.value is None
        assert result.coordinates["row"] == 0
        assert result.coordinates["column"] == "email"

    @pytest.mark.parametrize(
        ("row_index", "column", "expected_error", "description"),
        [
            (-1, "first_name", ToolError, "negative row index"),
            (10, "first_name", ToolError, "row index too large"),
            (0, "nonexistent", ColumnNotFoundError, "invalid column name"),
            (0, -1, ToolError, "negative column index"),
            (0, 20, ToolError, "column index too large"),
        ],
    )
    async def test_get_cell_value_boundary_errors(
        self,
        row_operations_ctx: Context,
        row_index: int,
        column: int | str,
        expected_error: type[Exception],
        description: str,
    ) -> None:
        """Test boundary condition error handling."""
        ctx = row_operations_ctx
        with pytest.raises(expected_error):
            get_cell_value(ctx, row_index, column)

    async def test_get_cell_value_invalid_session(self) -> None:
        """Test error handling for invalid session."""
        ctx = create_mock_context("invalid-session")
        with pytest.raises((ToolError, SessionNotFoundError, NoDataLoadedError)):
            get_cell_value(ctx, 0, "first_name")


@pytest.mark.asyncio
class TestSetCellValue:
    """Test set_cell_value server function."""

    @pytest.mark.parametrize(
        ("row_index", "column", "new_value", "description"),
        [
            (0, "first_name", "Johnny", "update string by column name"),
            (1, "age", 26, "update integer by column name"),
            (2, "is_active", True, "update boolean by column name"),
            (0, 0, 999, "update by column index"),
            (1, "salary", 57000, "update numeric value"),
            (0, "email", None, "set to null value"),
        ],
    )
    async def test_set_cell_value_success(
        self,
        row_operations_ctx: Context,
        row_index: int,
        column: int | str,
        new_value: CellValue,
        description: str,
    ) -> None:
        """Test successful cell value updates."""
        # Get original value first
        ctx = row_operations_ctx
        original = get_cell_value(ctx, row_index, column)

        # Update the value
        result = set_cell_value(ctx, row_index, column, new_value)

        assert result.success is True
        assert result.old_value == original.value
        assert result.new_value == new_value
        assert result.coordinates["row"] == row_index
        assert isinstance(result.coordinates["column"], str)

        # Verify the change persisted
        updated = get_cell_value(ctx, row_index, column)
        assert updated.value == new_value

    async def test_set_cell_value_type_conversion(self, row_operations_ctx: Context) -> None:
        """Test pandas type conversion handling."""
        # Set string to numeric column (pandas may preserve as string)
        ctx = row_operations_ctx
        result = set_cell_value(ctx, 0, "age", "35")

        assert result.success is True
        # Pandas behavior: string may be preserved in mixed-type column
        assert result.new_value in [35, "35"]  # Accept either conversion

        # Set numeric to string column
        result = set_cell_value(ctx, 0, "first_name", 42)

        assert result.success is True
        # Pandas will convert to string in object column
        assert str(result.new_value) == "42"

    @pytest.mark.parametrize(
        ("row_index", "column", "expected_error", "description"),
        [
            (-1, "first_name", ToolError, "negative row index"),
            (10, "first_name", ToolError, "row index too large"),
            (0, "nonexistent", ColumnNotFoundError, "invalid column name"),
            (0, -1, ToolError, "negative column index"),
            (0, 20, ToolError, "column index too large"),
        ],
    )
    async def test_set_cell_value_boundary_errors(
        self,
        row_operations_ctx: Context,
        row_index: int,
        column: int | str,
        expected_error: type[Exception],
        description: str,
    ) -> None:
        """Test boundary condition error handling."""
        ctx = row_operations_ctx
        with pytest.raises(expected_error):
            set_cell_value(ctx, row_index, column, "test")

    async def test_set_cell_value_invalid_session(self) -> None:
        """Test error handling for invalid session."""
        from databeak.exceptions import SessionNotFoundError

        ctx = create_mock_context("invalid-session")
        with pytest.raises((ToolError, SessionNotFoundError, NoDataLoadedError)):
            set_cell_value(ctx, 0, "first_name", "test")


@pytest.mark.asyncio
class TestGetRowData:
    """Test get_row_data server function."""

    async def test_get_row_data_all_columns(self, row_operations_ctx: Context) -> None:
        """Test retrieving all columns from a row."""
        ctx = row_operations_ctx
        result = get_row_data(ctx, 0)

        assert result.success is True
        assert result.row_index == 0
        assert len(result.data) == 8  # All columns
        assert "first_name" in result.data
        assert result.data["first_name"] == "John"
        assert result.data["age"] == 30
        assert len(result.columns) == 8

    async def test_get_row_data_selected_columns(self, row_operations_ctx: Context) -> None:
        """Test retrieving specific columns from a row."""
        columns = ["first_name", "last_name", "age"]
        ctx = row_operations_ctx
        result = get_row_data(ctx, 1, columns)

        assert result.success is True
        assert result.row_index == 1
        assert len(result.data) == 3
        assert result.data["first_name"] == "Jane"
        assert result.data["last_name"] == "Smith"
        assert result.data["age"] == 25
        assert result.columns == columns
        assert "email" not in result.data  # Not requested

    async def test_get_row_data_null_values(self, row_operations_ctx: Context) -> None:
        """Test row data retrieval with null values."""
        # Set some values to null first
        ctx = row_operations_ctx
        set_cell_value(ctx, 0, "email", None)
        set_cell_value(ctx, 0, "salary", None)

        result = get_row_data(ctx, 0)

        assert result.success is True
        assert result.data["email"] is None
        assert result.data["salary"] is None

    @pytest.mark.parametrize(
        ("row_index", "expected_error", "description"),
        [
            (-1, ToolError, "negative row index"),
            (10, ToolError, "row index too large"),
        ],
    )
    async def test_get_row_data_boundary_errors(
        self,
        row_operations_ctx: Context,
        row_index: int,
        expected_error: type[Exception],
        description: str,
    ) -> None:
        """Test boundary condition error handling."""
        ctx = row_operations_ctx
        with pytest.raises(expected_error):
            get_row_data(ctx, row_index)

    async def test_get_row_data_invalid_columns(self, row_operations_ctx: Context) -> None:
        """Test error handling for invalid column names."""
        ctx = row_operations_ctx
        with pytest.raises(ColumnNotFoundError):
            get_row_data(ctx, 0, ["first_name", "nonexistent"])

    async def test_get_row_data_invalid_session(self) -> None:
        """Test error handling for invalid session."""
        from databeak.exceptions import SessionNotFoundError

        ctx = create_mock_context("invalid-session")
        with pytest.raises((ToolError, SessionNotFoundError, NoDataLoadedError)):
            get_row_data(ctx, 0)


@pytest.mark.asyncio
class TestGetColumnData:
    """Test get_column_data server function."""

    async def test_get_column_data_full_column(self, row_operations_ctx: Context) -> None:
        """Test retrieving full column data."""
        ctx = row_operations_ctx
        result = get_column_data(ctx, "first_name")

        assert result.success is True
        assert result.column == "first_name"
        assert result.total_values == 4
        assert result.values == ["John", "Jane", "Bob", "Alice"]
        assert result.start_row == 0
        assert result.end_row == 3

    async def test_get_column_data_with_range(self, row_operations_ctx: Context) -> None:
        """Test retrieving column data with row range."""
        ctx = row_operations_ctx
        result = get_column_data(ctx, "age", 1, 3)

        assert result.success is True
        assert result.column == "age"
        assert result.total_values == 2
        assert result.values == [25, 35]  # Rows 1 and 2
        assert result.start_row == 1
        assert result.end_row == 3

    async def test_get_column_data_start_only(self, row_operations_ctx: Context) -> None:
        """Test retrieving column data from start row to end."""
        ctx = row_operations_ctx
        result = get_column_data(ctx, "last_name", 2)

        assert result.success is True
        assert result.column == "last_name"
        assert result.total_values == 2
        assert result.values == ["Johnson", "Brown"]  # Rows 2 and 3
        assert result.start_row == 2
        assert result.end_row == 3

    async def test_get_column_data_end_only(self, row_operations_ctx: Context) -> None:
        """Test retrieving column data from beginning to end row."""
        ctx = row_operations_ctx
        result = get_column_data(ctx, "email", None, 2)

        assert result.success is True
        assert result.column == "email"
        assert result.total_values == 2
        assert result.values == ["john@example.com", "jane@test.com"]  # Rows 0 and 1
        assert result.start_row == 0
        assert result.end_row == 2

    async def test_get_column_data_numeric_column(self, row_operations_ctx: Context) -> None:
        """Test retrieving numeric column data."""
        ctx = row_operations_ctx
        result = get_column_data(ctx, "salary")

        assert result.success is True
        assert result.values == [50000, 55000, 60000, 52000]
        assert all(isinstance(v, int) for v in result.values)

    async def test_get_column_data_boolean_column(self, row_operations_ctx: Context) -> None:
        """Test retrieving boolean column data."""
        ctx = row_operations_ctx
        result = get_column_data(ctx, "is_active")

        assert result.success is True
        assert result.values == [True, True, False, True]
        assert all(isinstance(v, bool) for v in result.values)

    @pytest.mark.parametrize(
        ("column", "start_row", "end_row", "expected_error", "description"),
        [
            ("nonexistent", None, None, ColumnNotFoundError, "invalid column name"),
            ("age", -1, None, InvalidParameterError, "negative start_row"),
            ("age", None, -1, InvalidParameterError, "negative end_row"),
            ("age", 10, None, ToolError, "start_row out of range"),
            ("age", None, 20, ToolError, "end_row out of range"),
            ("age", 3, 2, InvalidParameterError, "start_row >= end_row"),
        ],
    )
    async def test_get_column_data_boundary_errors(
        self,
        row_operations_ctx: Context,
        column: str,
        start_row: int | None,
        end_row: int | None,
        expected_error: type[Exception],
        description: str,
    ) -> None:
        """Test boundary condition error handling."""
        ctx = row_operations_ctx
        with pytest.raises(expected_error):
            get_column_data(ctx, column, start_row, end_row)

    async def test_get_column_data_invalid_session(self) -> None:
        """Test error handling for invalid session."""
        from databeak.exceptions import SessionNotFoundError

        ctx = create_mock_context("invalid-session")
        with pytest.raises((ToolError, SessionNotFoundError, NoDataLoadedError)):
            get_column_data(ctx, "first_name")


@pytest.mark.asyncio
class TestInsertRow:
    """Test insert_row server function."""

    async def test_insert_row_dict_format(self, row_operations_ctx: Context) -> None:
        """Test inserting row with dictionary format."""
        new_data: dict[str, CellValue] = {
            "id": 5,
            "first_name": "Charlie",
            "last_name": "Wilson",
            "age": 33,
            "email": "charlie@test.com",
            "salary": 58000,
            "is_active": True,
            "join_date": "2023-04-01",
        }

        ctx = row_operations_ctx
        result = insert_row(ctx, 2, new_data)

        assert result.success is True
        assert result.operation == "insert_row"
        assert result.row_index == 2
        assert result.rows_before == 4
        assert result.rows_after == 5
        assert result.data_inserted["first_name"] == "Charlie"
        assert len(result.columns) == 8

        # Verify insertion by checking the row
        row_result = get_row_data(ctx, 2)
        assert row_result.data["first_name"] == "Charlie"

    async def test_insert_row_list_format(self, row_operations_ctx: Context) -> None:
        """Test inserting row with list format."""
        new_data: list[CellValue] = [
            6,
            "Diana",
            "Davis",
            29,
            "diana@example.com",
            53000,
            True,
            "2023-05-01",
        ]

        ctx = row_operations_ctx
        result = insert_row(ctx, -1, new_data)  # Append at end

        assert result.success is True
        assert result.row_index == 4  # Should be last position
        assert result.rows_after == 5
        assert result.data_inserted["first_name"] == "Diana"

        # Verify insertion
        row_result = get_row_data(ctx, 4)
        assert row_result.data["first_name"] == "Diana"

    async def test_insert_row_json_string(self, row_operations_ctx: Context) -> None:
        """Test inserting row with JSON string format."""
        json_data = json.dumps(
            {
                "id": 7,
                "first_name": "Eve",
                "last_name": "Taylor",
                "age": 31,
                "email": None,  # Test null value
                "salary": 56000,
                "is_active": False,
                "join_date": "2023-06-01",
            },
        )

        ctx = row_operations_ctx
        result = insert_row(ctx, 0, json_data)  # Insert at beginning

        assert result.success is True
        assert result.row_index == 0
        assert result.data_inserted["first_name"] == "Eve"
        assert result.data_inserted["email"] is None

        # Verify insertion shifted other rows
        row_result = get_row_data(ctx, 1)
        assert row_result.data["first_name"] == "John"  # Original first row

    async def test_insert_row_partial_dict(self, row_operations_ctx: Context) -> None:
        """Test inserting row with partial dictionary (missing columns filled with None)."""
        partial_data: dict[str, CellValue] = {"first_name": "Frank", "age": 40, "is_active": True}

        ctx = row_operations_ctx
        result = insert_row(ctx, 1, partial_data)

        assert result.success is True
        assert result.data_inserted["first_name"] == "Frank"
        assert result.data_inserted["age"] == 40
        assert result.data_inserted["email"] is None  # Should be filled with None
        assert result.data_inserted["salary"] is None

    async def test_insert_row_null_values(self, row_operations_ctx: Context) -> None:
        """Test inserting row with explicit null values."""
        null_data: dict[str, CellValue] = {
            "id": 8,
            "first_name": "Grace",
            "last_name": None,
            "age": None,
            "email": "grace@test.com",
            "salary": None,
            "is_active": True,
            "join_date": None,
        }

        ctx = row_operations_ctx
        result = insert_row(ctx, 3, null_data)

        assert result.success is True
        assert result.data_inserted["first_name"] == "Grace"
        assert result.data_inserted["last_name"] is None
        assert result.data_inserted["age"] is None

        # Verify null values were preserved
        row_result = get_row_data(ctx, 3)
        assert row_result.data["last_name"] is None

    @pytest.mark.parametrize(
        ("row_index", "data", "expected_error", "description"),
        [
            (-2, {"first_name": "Test"}, ToolError, "invalid negative row index"),
            (10, {"first_name": "Test"}, ToolError, "row index too large"),
            (0, [1, 2], ToolError, "list length mismatch"),
            (0, '{"invalid": json}', ToolError, "invalid JSON string"),
            (0, 42, ToolError, "unsupported data type"),
        ],
    )
    async def test_insert_row_boundary_errors(
        self,
        row_operations_ctx: Context,
        row_index: int,
        data: dict[str, CellValue] | list[CellValue] | str | int,
        expected_error: type[Exception],
        description: str,
    ) -> None:
        """Test boundary condition error handling."""
        ctx = row_operations_ctx
        with pytest.raises(expected_error):
            insert_row(ctx, row_index, data)  # type: ignore[arg-type] # Testing invalid types

    async def test_insert_row_invalid_session(self) -> None:
        """Test error handling for invalid session."""
        from databeak.exceptions import SessionNotFoundError

        ctx = create_mock_context("invalid-session")
        with pytest.raises((ToolError, SessionNotFoundError, NoDataLoadedError)):
            insert_row(ctx, 0, {"first_name": "Test"})


@pytest.mark.asyncio
class TestDeleteRow:
    """Test delete_row server function."""

    async def test_delete_row_middle(self, row_operations_ctx: Context) -> None:
        """Test deleting a row from the middle."""
        # Get original data for verification
        ctx = row_operations_ctx
        original_row = get_row_data(ctx, 1)

        result = delete_row(ctx, 1)

        assert result.success is True
        assert result.operation == "delete_row"
        assert result.row_index == 1
        assert result.rows_before == 4
        assert result.rows_after == 3
        assert result.deleted_data["first_name"] == "Jane"
        assert result.deleted_data == original_row.data

        # Verify deletion shifted rows correctly
        new_row_1 = get_row_data(ctx, 1)
        assert new_row_1.data["first_name"] == "Bob"  # Was originally row 2

    async def test_delete_row_first(self, row_operations_ctx: Context) -> None:
        """Test deleting the first row."""
        ctx = row_operations_ctx
        result = delete_row(ctx, 0)

        assert result.success is True
        assert result.row_index == 0
        assert result.deleted_data["first_name"] == "John"

        # Verify first row is now what was second
        new_first = get_row_data(ctx, 0)
        assert new_first.data["first_name"] == "Jane"

    async def test_delete_row_last(self, row_operations_ctx: Context) -> None:
        """Test deleting the last row."""
        ctx = row_operations_ctx
        result = delete_row(ctx, 3)

        assert result.success is True
        assert result.row_index == 3
        assert result.deleted_data["first_name"] == "Alice"
        assert result.rows_after == 3

    async def test_delete_row_with_nulls(self, row_operations_ctx: Context) -> None:
        """Test deleting row with null values."""
        # Set some values to null first
        ctx = row_operations_ctx
        set_cell_value(ctx, 0, "email", None)
        set_cell_value(ctx, 0, "salary", None)

        result = delete_row(ctx, 0)

        assert result.success is True
        assert result.deleted_data["email"] is None
        assert result.deleted_data["salary"] is None

    @pytest.mark.parametrize(
        ("row_index", "expected_error", "description"),
        [
            (-1, ToolError, "negative row index"),
            (10, ToolError, "row index too large"),
        ],
    )
    async def test_delete_row_boundary_errors(
        self,
        row_operations_ctx: Context,
        row_index: int,
        expected_error: type[Exception],
        description: str,
    ) -> None:
        """Test boundary condition error handling."""
        ctx = row_operations_ctx
        with pytest.raises(expected_error):
            delete_row(ctx, row_index)

    async def test_delete_row_invalid_session(self) -> None:
        """Test error handling for invalid session."""
        from databeak.exceptions import SessionNotFoundError

        ctx = create_mock_context("invalid-session")
        with pytest.raises((ToolError, SessionNotFoundError, NoDataLoadedError)):
            delete_row(ctx, 0)


@pytest.mark.asyncio
class TestUpdateRow:
    """Test update_row server function."""

    async def test_update_row_multiple_columns(self, row_operations_ctx: Context) -> None:
        """Test updating multiple columns in a row."""
        updates: dict[str, CellValue] = {"age": 31, "salary": 51000, "is_active": False}

        ctx = row_operations_ctx
        result = update_row(ctx, 0, updates)

        assert result.success is True
        assert result.operation == "update_row"
        assert result.row_index == 0
        assert len(result.columns_updated) == 3
        assert "age" in result.columns_updated
        assert "salary" in result.columns_updated
        assert "is_active" in result.columns_updated
        assert result.changes_made == 3

        # Verify old values were captured
        assert result.old_values["age"] == 30
        assert result.old_values["salary"] == 50000
        assert result.old_values["is_active"] is True

        # Verify new values
        assert result.new_values["age"] == 31
        assert result.new_values["salary"] == 51000
        assert result.new_values["is_active"] is False

        # Verify changes persisted
        row_result = get_row_data(ctx, 0)
        assert row_result.data["age"] == 31
        assert row_result.data["salary"] == 51000
        assert row_result.data["is_active"] is False

    async def test_update_row_single_column(self, row_operations_ctx: Context) -> None:
        """Test updating a single column."""
        updates: dict[str, CellValue] = {"first_name": "Jonathan"}

        ctx = row_operations_ctx
        result = update_row(ctx, 0, updates)

        assert result.success is True
        assert result.columns_updated == ["first_name"]
        assert result.old_values["first_name"] == "John"
        assert result.new_values["first_name"] == "Jonathan"
        assert result.changes_made == 1

    async def test_update_row_with_nulls(self, row_operations_ctx: Context) -> None:
        """Test updating columns to null values."""
        updates: dict[str, CellValue] = {"email": None, "salary": None}

        ctx = row_operations_ctx
        result = update_row(ctx, 1, updates)

        assert result.success is True
        assert result.old_values["email"] == "jane@test.com"
        assert result.old_values["salary"] == 55000
        assert result.new_values["email"] is None
        assert result.new_values["salary"] is None

        # Verify nulls persisted
        row_result = get_row_data(ctx, 1)
        assert row_result.data["email"] is None
        assert row_result.data["salary"] is None

    async def test_update_row_json_string(self, row_operations_ctx: Context) -> None:
        """Test updating row with JSON string format."""
        json_updates = json.dumps({"last_name": "Smith-Jones", "age": 26})

        ctx = row_operations_ctx
        result = update_row(ctx, 1, json_updates)

        assert result.success is True
        assert result.columns_updated == ["last_name", "age"]
        assert result.new_values["last_name"] == "Smith-Jones"
        assert result.new_values["age"] == 26

    async def test_update_row_no_changes(self, row_operations_ctx: Context) -> None:
        """Test updating row with same values (no actual changes)."""
        # Get current values
        ctx = row_operations_ctx
        current = get_row_data(ctx, 0)
        current_age = current.data["age"]

        # "Update" with same value
        updates = {"age": current_age}

        result = update_row(ctx, 0, updates)

        assert result.success is True
        assert result.changes_made == 1  # Still counts as a change operation
        assert result.old_values["age"] == current_age
        assert result.new_values["age"] == current_age

    @pytest.mark.parametrize(
        ("row_index", "data", "expected_error", "description"),
        [
            (-1, {"age": 30}, ToolError, "negative row index"),
            (10, {"age": 30}, ToolError, "row index too large"),
            (0, {"nonexistent": "value"}, ColumnNotFoundError, "invalid column name"),
            (0, '{"invalid": json}', ToolError, "invalid JSON string"),
            (0, ["not", "dict"], ToolError, "non-dict data after JSON parsing"),
        ],
    )
    async def test_update_row_boundary_errors(
        self,
        row_operations_ctx: Context,
        row_index: int,
        data: dict[str, CellValue] | str,
        expected_error: type[Exception],
        description: str,
    ) -> None:
        """Test boundary condition error handling."""
        ctx = row_operations_ctx
        with pytest.raises(expected_error):
            update_row(ctx, row_index, data)

    async def test_update_row_invalid_session(self) -> None:
        """Test error handling for invalid session."""
        from databeak.exceptions import SessionNotFoundError

        ctx = create_mock_context("invalid-session")
        with pytest.raises((ToolError, SessionNotFoundError, NoDataLoadedError)):
            update_row(ctx, 0, {"age": 30})


@pytest.mark.asyncio
class TestRowOperationsIntegration:
    """Integration tests for combined row operations."""

    async def test_complete_row_lifecycle(self, row_operations_ctx: Context) -> None:
        """Test complete row manipulation workflow."""
        # 1. Get original data
        ctx = row_operations_ctx
        original = get_row_data(ctx, 0)
        assert original.data["first_name"] == "John"

        # 2. Update some values
        updates: dict[str, CellValue] = {"age": 32, "salary": 55000}
        update_result = update_row(ctx, 0, updates)
        assert update_result.success is True

        # 3. Insert a new row
        new_row: dict[str, CellValue] = {
            "id": 5,
            "first_name": "Test",
            "last_name": "User",
            "age": 25,
        }
        insert_result = insert_row(ctx, 1, new_row)
        assert insert_result.success is True

        # 4. Verify the insertion shifted indexes
        shifted_row = get_row_data(ctx, 2)  # Original row 1 is now at index 2
        assert shifted_row.data["first_name"] == "Jane"

        # 5. Delete the inserted row
        delete_result = delete_row(ctx, 1)
        assert delete_result.success is True
        assert delete_result.deleted_data["first_name"] == "Test"

        # 6. Verify indexes are back to normal
        restored_row = get_row_data(ctx, 1)
        assert restored_row.data["first_name"] == "Jane"

    async def test_cell_operations_consistency(self, row_operations_ctx: Context) -> None:
        """Test consistency between cell and row operations."""
        # Set a cell value
        ctx = row_operations_ctx
        set_result = set_cell_value(ctx, 0, "first_name", "Johnny")
        assert set_result.success is True

        # Verify with get_cell_value
        cell_result = get_cell_value(ctx, 0, "first_name")
        assert cell_result.value == "Johnny"

        # Verify with get_row_data
        row_result = get_row_data(ctx, 0)
        assert row_result.data["first_name"] == "Johnny"

        # Update multiple columns including the cell we just changed
        update_result = update_row(ctx, 0, {"first_name": "John", "age": 35})
        assert update_result.old_values["first_name"] == "Johnny"
        assert update_result.new_values["first_name"] == "John"

    async def test_column_data_after_modifications(self, row_operations_ctx: Context) -> None:
        """Test column data retrieval after row modifications."""
        # Get original column data
        ctx = row_operations_ctx
        original_names = get_column_data(ctx, "first_name")
        assert original_names.values == ["John", "Jane", "Bob", "Alice"]

        # Insert a row
        insert_row(ctx, 2, {"first_name": "Charlie", "age": 30})

        # Check column data reflects the change
        updated_names = get_column_data(ctx, "first_name")
        assert updated_names.values == ["John", "Jane", "Charlie", "Bob", "Alice"]

        # Delete a row
        delete_row(ctx, 1)  # Delete Jane

        # Check column data again
        final_names = get_column_data(ctx, "first_name")
        assert final_names.values == ["John", "Charlie", "Bob", "Alice"]

    async def test_boundary_condition_combinations(self, row_operations_ctx: Context) -> None:
        """Test edge cases with combined operations."""
        # Test operations on last row
        last_index = 3  # 4 rows, so last index is 3

        # Update last row
        ctx = row_operations_ctx
        update_result = update_row(ctx, last_index, {"age": 99})
        assert update_result.success is True

        # Get last row
        last_row = get_row_data(ctx, last_index)
        assert last_row.data["age"] == 99

        # Delete last row
        delete_result = delete_row(ctx, last_index)
        assert delete_result.success is True

        # Insert at new end (append)
        insert_result = insert_row(ctx, -1, {"first_name": "NewLast", "age": 50})
        assert insert_result.success is True

        # Verify new last row
        new_last = get_row_data(ctx, 3)  # Still 4 rows total
        assert new_last.data["first_name"] == "NewLast"
