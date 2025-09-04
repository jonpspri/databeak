"""Additional tests for transformations module to improve coverage."""

import pytest
import pandas as pd

from src.csv_editor.tools.transformations import (
    add_column,
    change_column_type,
    delete_row,
    extract_from_column,
    fill_column_nulls,
    fill_missing_values,
    filter_rows,
    get_cell_value,
    get_column_data,
    get_data_summary,
    get_row_data,
    insert_row,
    remove_columns,
    remove_duplicates,
    rename_columns,
    replace_in_column,
    select_columns,
    set_cell_value,
    sort_data,
    split_column,
    strip_column,
    transform_column_case,
    update_column,
    update_row,
)
from src.csv_editor.tools.io_operations import load_csv_from_content


@pytest.fixture
async def transform_test_session():
    """Create a session with transformation-friendly data."""
    csv_content = """name,age,email,phone,notes
John Doe,30,john@example.com,(555) 123-4567,Good customer
Jane Smith,25,jane@test.com,(555) 987-6543,
Bob Johnson,35,,555-111-2222,VIP client
Alice Brown,28,alice@company.org,(555) 444-5555,New customer"""

    result = await load_csv_from_content(csv_content)
    return result["session_id"]


@pytest.mark.asyncio
class TestTransformationErrorHandling:
    """Test transformation error handling paths."""

    async def test_filter_rows_invalid_operator(self, transform_test_session):
        """Test filtering with invalid operator."""
        result = await filter_rows(
            transform_test_session, [{"column": "age", "operator": "invalid_op", "value": 25}]
        )
        assert result["success"] is False
        assert "operator" in result["error"]["message"]

    async def test_filter_rows_missing_column(self, transform_test_session):
        """Test filtering with missing column."""
        result = await filter_rows(
            transform_test_session, [{"column": "nonexistent", "operator": "==", "value": "test"}]
        )
        assert result["success"] is False
        assert "not found" in result["error"]["message"]

    async def test_add_column_duplicate_name(self, transform_test_session):
        """Test adding column with existing name."""
        result = await add_column(transform_test_session, "name", "duplicate")
        assert result["success"] is False
        assert "already exists" in result["error"]

    async def test_add_column_invalid_formula(self, transform_test_session):
        """Test adding column with invalid formula."""
        result = await add_column(transform_test_session, "calculated", formula="invalid_syntax + ")
        assert result["success"] is False
        assert "Formula evaluation failed" in result["error"]

    async def test_add_column_list_length_mismatch(self, transform_test_session):
        """Test adding column with mismatched list length."""
        result = await add_column(
            transform_test_session, "new_col", value=[1, 2]
        )  # Only 2 values for 4 rows
        assert result["success"] is False
        assert "doesn't match row count" in result["error"]

    async def test_remove_columns_missing(self, transform_test_session):
        """Test removing non-existent columns."""
        result = await remove_columns(transform_test_session, ["nonexistent", "alsofake"])
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_change_column_type_invalid_type(self, transform_test_session):
        """Test changing to invalid data type."""
        result = await change_column_type(transform_test_session, "age", "invalid_type")
        assert result["success"] is False
        assert "Unsupported dtype" in result["error"]

    async def test_change_column_type_missing_column(self, transform_test_session):
        """Test changing type of missing column."""
        result = await change_column_type(transform_test_session, "nonexistent", "int")
        assert result["success"] is False
        assert "not found" in result["error"]


@pytest.mark.asyncio
class TestCellAndRowOperations:
    """Test cell and row operation edge cases."""

    async def test_get_cell_value_out_of_bounds(self, transform_test_session):
        """Test getting cell value with out of bounds indices."""
        # Test row out of bounds
        result = await get_cell_value(transform_test_session, 100, "name")
        assert result["success"] is False
        assert "out of range" in result["error"]

        # Test column out of bounds with index
        result = await get_cell_value(transform_test_session, 0, 100)
        assert result["success"] is False
        assert "out of range" in result["error"]

    async def test_set_cell_value_out_of_bounds(self, transform_test_session):
        """Test setting cell value with out of bounds indices."""
        result = await set_cell_value(transform_test_session, 100, "name", "New Value")
        assert result["success"] is False
        assert "out of range" in result["error"]

    async def test_get_row_data_out_of_bounds(self, transform_test_session):
        """Test getting row data with invalid index."""
        result = await get_row_data(transform_test_session, 100)
        assert result["success"] is False
        assert "out of range" in result["error"]

    async def test_get_row_data_invalid_columns(self, transform_test_session):
        """Test getting row data with invalid columns."""
        result = await get_row_data(transform_test_session, 0, ["nonexistent", "alsofake"])
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_insert_row_out_of_bounds(self, transform_test_session):
        """Test inserting row at invalid index."""
        result = await insert_row(transform_test_session, 100, {"name": "Test", "age": 25})
        assert result["success"] is False
        assert "out of range" in result["error"]

    async def test_insert_row_invalid_json(self, transform_test_session):
        """Test inserting row with invalid JSON string."""
        result = await insert_row(transform_test_session, 0, '{"invalid": json}')
        assert result["success"] is False
        assert "Invalid JSON" in result["error"]

    async def test_insert_row_wrong_data_type(self, transform_test_session):
        """Test inserting row with wrong data type."""
        result = await insert_row(transform_test_session, 0, 123)  # Integer instead of dict/list/string
        assert result["success"] is False
        assert "must be a dict or list" in result["error"]

    async def test_delete_row_out_of_bounds(self, transform_test_session):
        """Test deleting row with invalid index."""
        result = await delete_row(transform_test_session, 100)
        assert result["success"] is False
        assert "out of range" in result["error"]

    async def test_update_row_invalid_json(self, transform_test_session):
        """Test updating row with invalid JSON."""
        result = await update_row(transform_test_session, 0, '{"invalid": json}')
        assert result["success"] is False
        assert "Invalid JSON" in result["error"]

    async def test_update_row_invalid_columns(self, transform_test_session):
        """Test updating row with invalid columns."""
        result = await update_row(transform_test_session, 0, {"nonexistent": "value"})
        assert result["success"] is False
        assert "not found" in result["error"]


@pytest.mark.asyncio
class TestColumnOperations:
    """Test column operation edge cases."""

    async def test_get_column_data_invalid_column(self, transform_test_session):
        """Test getting column data for invalid column."""
        result = await get_column_data(transform_test_session, "nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_get_column_data_invalid_range(self, transform_test_session):
        """Test getting column data with invalid row range."""
        # Test invalid start row
        result = await get_column_data(transform_test_session, "name", start_row=100)
        assert result["success"] is False
        assert "out of range" in result["error"]

        # Test invalid end row
        result = await get_column_data(transform_test_session, "name", start_row=0, end_row=100)
        assert result["success"] is False
        assert "invalid" in result["error"]

    async def test_replace_in_column_missing_column(self, transform_test_session):
        """Test replacing in missing column."""
        result = await replace_in_column(
            transform_test_session, "nonexistent", "pattern", "replacement"
        )
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_extract_from_column_missing_column(self, transform_test_session):
        """Test extracting from missing column."""
        result = await extract_from_column(transform_test_session, "nonexistent", r"(\w+)")
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_split_column_missing_column(self, transform_test_session):
        """Test splitting missing column."""
        result = await split_column(transform_test_session, "nonexistent", " ")
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_transform_column_case_invalid_transform(self, transform_test_session):
        """Test case transformation with invalid transform type."""
        result = await transform_column_case(transform_test_session, "name", "invalid")
        assert result["success"] is False
        assert "Unknown transform" in result["error"]

    async def test_fill_column_nulls_missing_column(self, transform_test_session):
        """Test filling nulls in missing column."""
        result = await fill_column_nulls(transform_test_session, "nonexistent", "value")
        assert result["success"] is False
        assert "not found" in result["error"]


@pytest.mark.asyncio
class TestDataSummaryAndInspection:
    """Test data summary and inspection functions."""

    async def test_get_data_summary_without_preview(self, transform_test_session):
        """Test data summary without preview."""
        result = await get_data_summary(transform_test_session, include_preview=False)
        assert result["success"] is True
        assert "preview" not in result
        assert "coordinate_system" in result

    async def test_get_data_summary_custom_preview_size(self, transform_test_session):
        """Test data summary with custom preview size."""
        result = await get_data_summary(
            transform_test_session, include_preview=True, max_preview_rows=2
        )
        assert result["success"] is True
        assert "preview" in result
        assert result["preview"]["preview_rows"] <= 2


@pytest.mark.asyncio
class TestAdvancedTransformations:
    """Test advanced transformation features."""

    async def test_fill_missing_values_different_strategies(self, transform_test_session):
        """Test different missing value fill strategies."""
        strategies = ["drop", "forward", "backward", "mean", "median", "mode"]

        for strategy in strategies:
            result = await fill_missing_values(transform_test_session, strategy=strategy)
            # Some strategies might fail depending on data types, but should handle gracefully
            assert "success" in result

    async def test_fill_missing_values_invalid_strategy(self, transform_test_session):
        """Test fill missing values with invalid strategy."""
        result = await fill_missing_values(transform_test_session, strategy="invalid")
        assert result["success"] is False
        assert "Unknown strategy" in result["error"]

    async def test_remove_duplicates_different_options(self, transform_test_session):
        """Test duplicate removal with different options."""
        # First add some duplicate data
        await insert_row(
            transform_test_session, -1, {"name": "John Doe", "age": 30, "email": "john@example.com"}
        )

        # Test different keep options
        for keep_option in ["first", "last", "none"]:
            result = await remove_duplicates(transform_test_session, keep=keep_option)
            assert result["success"] is True
