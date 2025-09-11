"""Unit tests for row operations server module.

Tests the server wrapper layer, parameter validation, and FastMCP integration for row-level
operations with comprehensive boundary checking and error handling.
"""

import json
import pytest
from fastmcp.exceptions import ToolError

# Ensure full module coverage
import src.databeak.servers.row_operations_server  # noqa: F401
from src.databeak.exceptions import ColumnNotFoundError, InvalidParameterError
from src.databeak.servers.io_server import load_csv_from_content
from src.databeak.servers.row_operations_server import (
    delete_row,
    get_cell_value,
    get_column_data,
    get_row_data,
    insert_row,
    set_cell_value,
    update_row,
)


@pytest.fixture
async def row_operations_session():
    """Create a test session with row operations test data."""
    csv_content = """id,first_name,last_name,age,email,salary,is_active,join_date
1,John,Doe,30,john@example.com,50000,true,2023-01-15
2,Jane,Smith,25,jane@test.com,55000,true,2023-02-01
3,Bob,Johnson,35,bob@company.org,60000,false,2023-01-10
4,Alice,Brown,28,alice@example.com,52000,true,2023-03-01"""

    result = await load_csv_from_content(csv_content)
    return result.session_id


@pytest.mark.asyncio
class TestGetCellValue:
    """Test get_cell_value server function."""

    @pytest.mark.parametrize(
        "row_index,column,expected_value,description",
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
        self, row_operations_session, row_index, column, expected_value, description
    ):
        """Test successful cell value retrieval with various coordinates."""
        result = get_cell_value(row_operations_session, row_index, column)

        assert result.success is True
        assert result.value == expected_value
        assert result.coordinates["row"] == row_index
        assert isinstance(result.coordinates["column"], str)  # Always returns column name
        assert result.data_type is not None

    async def test_get_cell_value_null_handling(self, row_operations_session):
        """Test null value handling in cell retrieval."""
        # First set a cell to None to test null handling
        set_cell_value(row_operations_session, 0, "email", None)

        result = get_cell_value(row_operations_session, 0, "email")

        assert result.success is True
        assert result.value is None
        assert result.coordinates["row"] == 0
        assert result.coordinates["column"] == "email"

    @pytest.mark.parametrize(
        "row_index,column,expected_error,description",
        [
            (-1, "first_name", ToolError, "negative row index"),
            (10, "first_name", ToolError, "row index too large"),
            (0, "nonexistent", ColumnNotFoundError, "invalid column name"),
            (0, -1, ToolError, "negative column index"),
            (0, 20, ToolError, "column index too large"),
        ],
    )
    async def test_get_cell_value_boundary_errors(
        self, row_operations_session, row_index, column, expected_error, description
    ):
        """Test boundary condition error handling."""
        with pytest.raises(expected_error):
            get_cell_value(row_operations_session, row_index, column)

    async def test_get_cell_value_invalid_session(self):
        """Test error handling for invalid session."""
        with pytest.raises(ToolError):
            get_cell_value("invalid-session", 0, "first_name")


@pytest.mark.asyncio
class TestSetCellValue:
    """Test set_cell_value server function."""

    @pytest.mark.parametrize(
        "row_index,column,new_value,description",
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
        self, row_operations_session, row_index, column, new_value, description
    ):
        """Test successful cell value updates."""
        # Get original value first
        original = get_cell_value(row_operations_session, row_index, column)

        # Update the value
        result = set_cell_value(row_operations_session, row_index, column, new_value)

        assert result.success is True
        assert result.old_value == original.value
        assert result.new_value == new_value
        assert result.coordinates["row"] == row_index
        assert isinstance(result.coordinates["column"], str)

        # Verify the change persisted
        updated = get_cell_value(row_operations_session, row_index, column)
        assert updated.value == new_value

    async def test_set_cell_value_type_conversion(self, row_operations_session):
        """Test pandas type conversion handling."""
        # Set string to numeric column (pandas may preserve as string)
        result = set_cell_value(row_operations_session, 0, "age", "35")

        assert result.success is True
        # Pandas behavior: string may be preserved in mixed-type column
        assert result.new_value in [35, "35"]  # Accept either conversion

        # Set numeric to string column
        result = set_cell_value(row_operations_session, 0, "first_name", 42)

        assert result.success is True
        # Pandas will convert to string in object column
        assert str(result.new_value) == "42"

    @pytest.mark.parametrize(
        "row_index,column,expected_error,description",
        [
            (-1, "first_name", ToolError, "negative row index"),
            (10, "first_name", ToolError, "row index too large"),
            (0, "nonexistent", ColumnNotFoundError, "invalid column name"),
            (0, -1, ToolError, "negative column index"),
            (0, 20, ToolError, "column index too large"),
        ],
    )
    async def test_set_cell_value_boundary_errors(
        self, row_operations_session, row_index, column, expected_error, description
    ):
        """Test boundary condition error handling."""
        with pytest.raises(expected_error):
            set_cell_value(row_operations_session, row_index, column, "test")

    async def test_set_cell_value_invalid_session(self):
        """Test error handling for invalid session."""
        with pytest.raises(ToolError):
            set_cell_value("invalid-session", 0, "first_name", "test")


@pytest.mark.asyncio
class TestGetRowData:
    """Test get_row_data server function."""

    async def test_get_row_data_all_columns(self, row_operations_session):
        """Test retrieving all columns from a row."""
        result = get_row_data(row_operations_session, 0)

        assert result.success is True
        assert result.session_id == row_operations_session
        assert result.row_index == 0
        assert len(result.data) == 8  # All columns
        assert "first_name" in result.data
        assert result.data["first_name"] == "John"
        assert result.data["age"] == 30
        assert len(result.columns) == 8

    async def test_get_row_data_selected_columns(self, row_operations_session):
        """Test retrieving specific columns from a row."""
        columns = ["first_name", "last_name", "age"]
        result = get_row_data(row_operations_session, 1, columns)

        assert result.success is True
        assert result.row_index == 1
        assert len(result.data) == 3
        assert result.data["first_name"] == "Jane"
        assert result.data["last_name"] == "Smith"
        assert result.data["age"] == 25
        assert result.columns == columns
        assert "email" not in result.data  # Not requested

    async def test_get_row_data_null_values(self, row_operations_session):
        """Test row data retrieval with null values."""
        # Set some values to null first
        set_cell_value(row_operations_session, 0, "email", None)
        set_cell_value(row_operations_session, 0, "salary", None)

        result = get_row_data(row_operations_session, 0)

        assert result.success is True
        assert result.data["email"] is None
        assert result.data["salary"] is None

    @pytest.mark.parametrize(
        "row_index,expected_error,description",
        [
            (-1, ToolError, "negative row index"),
            (10, ToolError, "row index too large"),
        ],
    )
    async def test_get_row_data_boundary_errors(
        self, row_operations_session, row_index, expected_error, description
    ):
        """Test boundary condition error handling."""
        with pytest.raises(expected_error):
            get_row_data(row_operations_session, row_index)

    async def test_get_row_data_invalid_columns(self, row_operations_session):
        """Test error handling for invalid column names."""
        with pytest.raises(ColumnNotFoundError):
            get_row_data(row_operations_session, 0, ["first_name", "nonexistent"])

    async def test_get_row_data_invalid_session(self):
        """Test error handling for invalid session."""
        with pytest.raises(ToolError):
            get_row_data("invalid-session", 0)


@pytest.mark.asyncio
class TestGetColumnData:
    """Test get_column_data server function."""

    async def test_get_column_data_full_column(self, row_operations_session):
        """Test retrieving full column data."""
        result = get_column_data(row_operations_session, "first_name")

        assert result.success is True
        assert result.session_id == row_operations_session
        assert result.column == "first_name"
        assert result.total_values == 4
        assert result.values == ["John", "Jane", "Bob", "Alice"]
        assert result.start_row is None
        assert result.end_row is None

    async def test_get_column_data_with_range(self, row_operations_session):
        """Test retrieving column data with row range."""
        result = get_column_data(row_operations_session, "age", 1, 3)

        assert result.success is True
        assert result.column == "age"
        assert result.total_values == 2
        assert result.values == [25, 35]  # Rows 1 and 2
        assert result.start_row == 1
        assert result.end_row == 3

    async def test_get_column_data_start_only(self, row_operations_session):
        """Test retrieving column data from start row to end."""
        result = get_column_data(row_operations_session, "last_name", 2)

        assert result.success is True
        assert result.column == "last_name"
        assert result.total_values == 2
        assert result.values == ["Johnson", "Brown"]  # Rows 2 and 3
        assert result.start_row == 2
        assert result.end_row is None

    async def test_get_column_data_end_only(self, row_operations_session):
        """Test retrieving column data from beginning to end row."""
        result = get_column_data(row_operations_session, "email", None, 2)

        assert result.success is True
        assert result.column == "email"
        assert result.total_values == 2
        assert result.values == ["john@example.com", "jane@test.com"]  # Rows 0 and 1
        assert result.start_row is None
        assert result.end_row == 2

    async def test_get_column_data_numeric_column(self, row_operations_session):
        """Test retrieving numeric column data."""
        result = get_column_data(row_operations_session, "salary")

        assert result.success is True
        assert result.values == [50000, 55000, 60000, 52000]
        assert all(isinstance(v, int) for v in result.values)

    async def test_get_column_data_boolean_column(self, row_operations_session):
        """Test retrieving boolean column data."""
        result = get_column_data(row_operations_session, "is_active")

        assert result.success is True
        assert result.values == [True, True, False, True]
        assert all(isinstance(v, bool) for v in result.values)

    @pytest.mark.parametrize(
        "column,start_row,end_row,expected_error,description",
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
        self, row_operations_session, column, start_row, end_row, expected_error, description
    ):
        """Test boundary condition error handling."""
        with pytest.raises(expected_error):
            get_column_data(row_operations_session, column, start_row, end_row)

    async def test_get_column_data_invalid_session(self):
        """Test error handling for invalid session."""
        with pytest.raises(ToolError):
            get_column_data("invalid-session", "first_name")


@pytest.mark.asyncio
class TestInsertRow:
    """Test insert_row server function."""

    async def test_insert_row_dict_format(self, row_operations_session):
        """Test inserting row with dictionary format."""
        new_data = {
            "id": 5,
            "first_name": "Charlie",
            "last_name": "Wilson",
            "age": 33,
            "email": "charlie@test.com",
            "salary": 58000,
            "is_active": True,
            "join_date": "2023-04-01",
        }

        result = insert_row(row_operations_session, 2, new_data)

        assert result.success is True
        assert result.operation == "insert_row"
        assert result.row_index == 2
        assert result.rows_before == 4
        assert result.rows_after == 5
        assert result.data_inserted["first_name"] == "Charlie"
        assert len(result.columns) == 8

        # Verify insertion by checking the row
        row_result = get_row_data(row_operations_session, 2)
        assert row_result.data["first_name"] == "Charlie"

    async def test_insert_row_list_format(self, row_operations_session):
        """Test inserting row with list format."""
        new_data = [6, "Diana", "Davis", 29, "diana@example.com", 53000, True, "2023-05-01"]

        result = insert_row(row_operations_session, -1, new_data)  # Append at end

        assert result.success is True
        assert result.row_index == 4  # Should be last position
        assert result.rows_after == 5
        assert result.data_inserted["first_name"] == "Diana"

        # Verify insertion
        row_result = get_row_data(row_operations_session, 4)
        assert row_result.data["first_name"] == "Diana"

    async def test_insert_row_json_string(self, row_operations_session):
        """Test inserting row with JSON string format."""
        json_data = json.dumps({
            "id": 7,
            "first_name": "Eve",
            "last_name": "Taylor",
            "age": 31,
            "email": None,  # Test null value
            "salary": 56000,
            "is_active": False,
            "join_date": "2023-06-01",
        })

        result = insert_row(row_operations_session, 0, json_data)  # Insert at beginning

        assert result.success is True
        assert result.row_index == 0
        assert result.data_inserted["first_name"] == "Eve"
        assert result.data_inserted["email"] is None

        # Verify insertion shifted other rows
        row_result = get_row_data(row_operations_session, 1)
        assert row_result.data["first_name"] == "John"  # Original first row

    async def test_insert_row_partial_dict(self, row_operations_session):
        """Test inserting row with partial dictionary (missing columns filled with None)."""
        partial_data = {"first_name": "Frank", "age": 40, "is_active": True}

        result = insert_row(row_operations_session, 1, partial_data)

        assert result.success is True
        assert result.data_inserted["first_name"] == "Frank"
        assert result.data_inserted["age"] == 40
        assert result.data_inserted["email"] is None  # Should be filled with None
        assert result.data_inserted["salary"] is None

    async def test_insert_row_null_values(self, row_operations_session):
        """Test inserting row with explicit null values."""
        null_data = {
            "id": 8,
            "first_name": "Grace",
            "last_name": None,
            "age": None,
            "email": "grace@test.com",
            "salary": None,
            "is_active": True,
            "join_date": None,
        }

        result = insert_row(row_operations_session, 3, null_data)

        assert result.success is True
        assert result.data_inserted["first_name"] == "Grace"
        assert result.data_inserted["last_name"] is None
        assert result.data_inserted["age"] is None

        # Verify null values were preserved
        row_result = get_row_data(row_operations_session, 3)
        assert row_result.data["last_name"] is None

    @pytest.mark.parametrize(
        "row_index,data,expected_error,description",
        [
            (-2, {"first_name": "Test"}, ToolError, "invalid negative row index"),
            (10, {"first_name": "Test"}, ToolError, "row index too large"),
            (0, [1, 2], ToolError, "list length mismatch"),
            (0, '{"invalid": json}', ToolError, "invalid JSON string"),
            (0, 42, ToolError, "unsupported data type"),
        ],
    )
    async def test_insert_row_boundary_errors(
        self, row_operations_session, row_index, data, expected_error, description
    ):
        """Test boundary condition error handling."""
        with pytest.raises(expected_error):
            insert_row(row_operations_session, row_index, data)

    async def test_insert_row_invalid_session(self):
        """Test error handling for invalid session."""
        with pytest.raises(ToolError):
            insert_row("invalid-session", 0, {"first_name": "Test"})


@pytest.mark.asyncio
class TestDeleteRow:
    """Test delete_row server function."""

    async def test_delete_row_middle(self, row_operations_session):
        """Test deleting a row from the middle."""
        # Get original data for verification
        original_row = get_row_data(row_operations_session, 1)

        result = delete_row(row_operations_session, 1)

        assert result.success is True
        assert result.operation == "delete_row"
        assert result.row_index == 1
        assert result.rows_before == 4
        assert result.rows_after == 3
        assert result.deleted_data["first_name"] == "Jane"
        assert result.deleted_data == original_row.data

        # Verify deletion shifted rows correctly
        new_row_1 = get_row_data(row_operations_session, 1)
        assert new_row_1.data["first_name"] == "Bob"  # Was originally row 2

    async def test_delete_row_first(self, row_operations_session):
        """Test deleting the first row."""
        result = delete_row(row_operations_session, 0)

        assert result.success is True
        assert result.row_index == 0
        assert result.deleted_data["first_name"] == "John"

        # Verify first row is now what was second
        new_first = get_row_data(row_operations_session, 0)
        assert new_first.data["first_name"] == "Jane"

    async def test_delete_row_last(self, row_operations_session):
        """Test deleting the last row."""
        result = delete_row(row_operations_session, 3)

        assert result.success is True
        assert result.row_index == 3
        assert result.deleted_data["first_name"] == "Alice"
        assert result.rows_after == 3

    async def test_delete_row_with_nulls(self, row_operations_session):
        """Test deleting row with null values."""
        # Set some values to null first
        set_cell_value(row_operations_session, 0, "email", None)
        set_cell_value(row_operations_session, 0, "salary", None)

        result = delete_row(row_operations_session, 0)

        assert result.success is True
        assert result.deleted_data["email"] is None
        assert result.deleted_data["salary"] is None

    @pytest.mark.parametrize(
        "row_index,expected_error,description",
        [
            (-1, ToolError, "negative row index"),
            (10, ToolError, "row index too large"),
        ],
    )
    async def test_delete_row_boundary_errors(
        self, row_operations_session, row_index, expected_error, description
    ):
        """Test boundary condition error handling."""
        with pytest.raises(expected_error):
            delete_row(row_operations_session, row_index)

    async def test_delete_row_invalid_session(self):
        """Test error handling for invalid session."""
        with pytest.raises(ToolError):
            delete_row("invalid-session", 0)


@pytest.mark.asyncio
class TestUpdateRow:
    """Test update_row server function."""

    async def test_update_row_multiple_columns(self, row_operations_session):
        """Test updating multiple columns in a row."""
        updates = {"age": 31, "salary": 51000, "is_active": False}

        result = update_row(row_operations_session, 0, updates)

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
        row_result = get_row_data(row_operations_session, 0)
        assert row_result.data["age"] == 31
        assert row_result.data["salary"] == 51000
        assert row_result.data["is_active"] is False

    async def test_update_row_single_column(self, row_operations_session):
        """Test updating a single column."""
        updates = {"first_name": "Jonathan"}

        result = update_row(row_operations_session, 0, updates)

        assert result.success is True
        assert result.columns_updated == ["first_name"]
        assert result.old_values["first_name"] == "John"
        assert result.new_values["first_name"] == "Jonathan"
        assert result.changes_made == 1

    async def test_update_row_with_nulls(self, row_operations_session):
        """Test updating columns to null values."""
        updates = {"email": None, "salary": None}

        result = update_row(row_operations_session, 1, updates)

        assert result.success is True
        assert result.old_values["email"] == "jane@test.com"
        assert result.old_values["salary"] == 55000
        assert result.new_values["email"] is None
        assert result.new_values["salary"] is None

        # Verify nulls persisted
        row_result = get_row_data(row_operations_session, 1)
        assert row_result.data["email"] is None
        assert row_result.data["salary"] is None

    async def test_update_row_json_string(self, row_operations_session):
        """Test updating row with JSON string format."""
        json_updates = json.dumps({"last_name": "Smith-Jones", "age": 26})

        result = update_row(row_operations_session, 1, json_updates)

        assert result.success is True
        assert result.columns_updated == ["last_name", "age"]
        assert result.new_values["last_name"] == "Smith-Jones"
        assert result.new_values["age"] == 26

    async def test_update_row_no_changes(self, row_operations_session):
        """Test updating row with same values (no actual changes)."""
        # Get current values
        current = get_row_data(row_operations_session, 0)
        current_age = current.data["age"]

        # "Update" with same value
        updates = {"age": current_age}

        result = update_row(row_operations_session, 0, updates)

        assert result.success is True
        assert result.changes_made == 1  # Still counts as a change operation
        assert result.old_values["age"] == current_age
        assert result.new_values["age"] == current_age

    @pytest.mark.parametrize(
        "row_index,data,expected_error,description",
        [
            (-1, {"age": 30}, ToolError, "negative row index"),
            (10, {"age": 30}, ToolError, "row index too large"),
            (0, {"nonexistent": "value"}, ColumnNotFoundError, "invalid column name"),
            (0, '{"invalid": json}', ToolError, "invalid JSON string"),
            (0, ["not", "dict"], ToolError, "non-dict data after JSON parsing"),
        ],
    )
    async def test_update_row_boundary_errors(
        self, row_operations_session, row_index, data, expected_error, description
    ):
        """Test boundary condition error handling."""
        with pytest.raises(expected_error):
            update_row(row_operations_session, row_index, data)

    async def test_update_row_invalid_session(self):
        """Test error handling for invalid session."""
        with pytest.raises(ToolError):
            update_row("invalid-session", 0, {"age": 30})


@pytest.mark.asyncio
class TestRowOperationsIntegration:
    """Integration tests for combined row operations."""

    async def test_complete_row_lifecycle(self, row_operations_session):
        """Test complete row manipulation workflow."""
        # 1. Get original data
        original = get_row_data(row_operations_session, 0)
        assert original.data["first_name"] == "John"

        # 2. Update some values
        updates = {"age": 32, "salary": 55000}
        update_result = update_row(row_operations_session, 0, updates)
        assert update_result.success is True

        # 3. Insert a new row
        new_row = {"id": 5, "first_name": "Test", "last_name": "User", "age": 25}
        insert_result = insert_row(row_operations_session, 1, new_row)
        assert insert_result.success is True

        # 4. Verify the insertion shifted indexes
        shifted_row = get_row_data(row_operations_session, 2)  # Original row 1 is now at index 2
        assert shifted_row.data["first_name"] == "Jane"

        # 5. Delete the inserted row
        delete_result = delete_row(row_operations_session, 1)
        assert delete_result.success is True
        assert delete_result.deleted_data["first_name"] == "Test"

        # 6. Verify indexes are back to normal
        restored_row = get_row_data(row_operations_session, 1)
        assert restored_row.data["first_name"] == "Jane"

    async def test_cell_operations_consistency(self, row_operations_session):
        """Test consistency between cell and row operations."""
        # Set a cell value
        set_result = set_cell_value(row_operations_session, 0, "first_name", "Johnny")
        assert set_result.success is True

        # Verify with get_cell_value
        cell_result = get_cell_value(row_operations_session, 0, "first_name")
        assert cell_result.value == "Johnny"

        # Verify with get_row_data
        row_result = get_row_data(row_operations_session, 0)
        assert row_result.data["first_name"] == "Johnny"

        # Update multiple columns including the cell we just changed
        update_result = update_row(row_operations_session, 0, {"first_name": "John", "age": 35})
        assert update_result.old_values["first_name"] == "Johnny"
        assert update_result.new_values["first_name"] == "John"

    async def test_column_data_after_modifications(self, row_operations_session):
        """Test column data retrieval after row modifications."""
        # Get original column data
        original_names = get_column_data(row_operations_session, "first_name")
        assert original_names.values == ["John", "Jane", "Bob", "Alice"]

        # Insert a row
        insert_row(row_operations_session, 2, {"first_name": "Charlie", "age": 30})

        # Check column data reflects the change
        updated_names = get_column_data(row_operations_session, "first_name")
        assert updated_names.values == ["John", "Jane", "Charlie", "Bob", "Alice"]

        # Delete a row
        delete_row(row_operations_session, 1)  # Delete Jane

        # Check column data again
        final_names = get_column_data(row_operations_session, "first_name")
        assert final_names.values == ["John", "Charlie", "Bob", "Alice"]

    async def test_boundary_condition_combinations(self, row_operations_session):
        """Test edge cases with combined operations."""
        # Test operations on last row
        last_index = 3  # 4 rows, so last index is 3

        # Update last row
        update_result = update_row(row_operations_session, last_index, {"age": 99})
        assert update_result.success is True

        # Get last row
        last_row = get_row_data(row_operations_session, last_index)
        assert last_row.data["age"] == 99

        # Delete last row
        delete_result = delete_row(row_operations_session, last_index)
        assert delete_result.success is True

        # Insert at new end (append)
        insert_result = insert_row(row_operations_session, -1, {"first_name": "NewLast", "age": 50})
        assert insert_result.success is True

        # Verify new last row
        new_last = get_row_data(row_operations_session, 3)  # Still 4 rows total
        assert new_last.data["first_name"] == "NewLast"