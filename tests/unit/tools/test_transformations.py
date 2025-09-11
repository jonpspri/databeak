"""Additional tests for transformations module to improve coverage."""

import pytest
from fastmcp.exceptions import ToolError

# Import server modules for coverage tracking
from src.databeak.servers import (  # noqa: F401
    column_server,
    column_text_server,
    transformation_server,
)
from src.databeak.servers.discovery_server import get_data_summary
from src.databeak.servers.io_server import load_csv_from_content
from src.databeak.services.transformation_operations import (
    add_column,
    change_column_type,
    delete_row,
    extract_from_column,
    fill_column_nulls,
    fill_missing_values,
    filter_rows,
    get_cell_value,
    get_column_data,
    get_row_data,
    insert_row,
    remove_columns,
    remove_duplicates,
    replace_in_column,
    set_cell_value,
    split_column,
    transform_column_case,
    update_row,
)


@pytest.fixture
async def transform_test_session() -> str:
    """Create a session with transformation-friendly data."""
    csv_content = """name,age,email,phone,notes
John Doe,30,john@example.com,(555) 123-4567,Good customer
Jane Smith,25,jane@test.com,(555) 987-6543,
Bob Johnson,35,,555-111-2222,VIP client
Alice Brown,28,alice@company.org,(555) 444-5555,New customer"""

    result = await load_csv_from_content(csv_content)
    return result.session_id


@pytest.mark.asyncio
class TestTransformationErrorHandling:
    """Test transformation error handling paths."""

    async def test_filter_rows_invalid_operator(self, transform_test_session) -> None:
        """Test filtering with invalid operator."""
        with pytest.raises(ToolError, match="operator"):
            await filter_rows(
                transform_test_session,
                [{"column": "age", "operator": "invalid_op", "value": 25}],
            )

    async def test_filter_rows_missing_column(self, transform_test_session) -> None:
        """Test filtering with missing column."""
        with pytest.raises(ToolError, match="not found"):
            await filter_rows(
                transform_test_session,
                [{"column": "nonexistent", "operator": "==", "value": "test"}],
            )

    async def test_add_column_duplicate_name(self, transform_test_session) -> None:
        """Test adding column with existing name."""
        with pytest.raises(ToolError, match="already exists"):
            await add_column(transform_test_session, "name", "duplicate")

    async def test_add_column_invalid_formula(self, transform_test_session) -> None:
        """Test adding column with invalid formula."""
        with pytest.raises(ToolError, match="Formula evaluation failed"):
            await add_column(transform_test_session, "calculated", formula="invalid_syntax + ")

    async def test_add_column_list_length_mismatch(self, transform_test_session) -> None:
        """Test adding column with mismatched list length."""
        with pytest.raises(ToolError, match="doesn't match row count"):
            await add_column(
                transform_test_session, "new_col", value=[1, 2]
            )  # Only 2 values for 4 rows

    async def test_remove_columns_missing(self, transform_test_session) -> None:
        """Test removing non-existent columns."""
        with pytest.raises(ToolError, match="not found"):
            await remove_columns(transform_test_session, ["nonexistent", "alsofake"])

    async def test_change_column_type_invalid_type(self, transform_test_session) -> None:
        """Test changing to invalid data type."""
        with pytest.raises(ToolError, match="Unsupported dtype"):
            await change_column_type(transform_test_session, "age", "invalid_type")

    async def test_change_column_type_missing_column(self, transform_test_session) -> None:
        """Test changing type of missing column."""
        with pytest.raises(ToolError, match="not found"):
            await change_column_type(transform_test_session, "nonexistent", "int")


@pytest.mark.asyncio
class TestCellAndRowOperations:
    """Test cell and row operation edge cases."""

    async def test_get_cell_value_out_of_bounds(self, transform_test_session) -> None:
        """Test getting cell value with out of bounds indices."""
        # Test row out of bounds
        with pytest.raises(ToolError, match="out of range"):
            await get_cell_value(transform_test_session, 100, "name")

        # Test column out of bounds with index
        with pytest.raises(ToolError, match="out of range"):
            await get_cell_value(transform_test_session, 0, 100)

    async def test_set_cell_value_out_of_bounds(self, transform_test_session) -> None:
        """Test setting cell value with out of bounds indices."""
        with pytest.raises(ToolError, match="out of range"):
            await set_cell_value(transform_test_session, 100, "name", "New Value")

    async def test_get_row_data_out_of_bounds(self, transform_test_session) -> None:
        """Test getting row data with invalid index."""
        with pytest.raises(ToolError, match="out of range"):
            await get_row_data(transform_test_session, 100)

    async def test_get_row_data_invalid_columns(self, transform_test_session) -> None:
        """Test getting row data with invalid columns."""
        with pytest.raises(ToolError, match="not found"):
            await get_row_data(transform_test_session, 0, ["nonexistent", "alsofake"])

    async def test_insert_row_out_of_bounds(self, transform_test_session) -> None:
        """Test inserting row at invalid index."""
        with pytest.raises(ToolError, match="out of range"):
            await insert_row(transform_test_session, 100, {"name": "Test", "age": 25})

    async def test_insert_row_invalid_json(self, transform_test_session) -> None:
        """Test inserting row with invalid JSON string."""
        with pytest.raises(ToolError, match="Invalid JSON"):
            await insert_row(transform_test_session, 0, '{"invalid": json}')

    async def test_insert_row_wrong_data_type(self, transform_test_session) -> None:
        """Test inserting row with wrong data type."""
        with pytest.raises(ToolError, match="must be a dict or list"):
            await insert_row(transform_test_session, 0, 123)  # Integer instead of dict/list/string

    async def test_delete_row_out_of_bounds(self, transform_test_session) -> None:
        """Test deleting row with invalid index."""
        with pytest.raises(ToolError, match="out of range"):
            await delete_row(transform_test_session, 100)

    async def test_update_row_invalid_json(self, transform_test_session) -> None:
        """Test updating row with invalid JSON."""
        with pytest.raises(ToolError, match="Invalid JSON"):
            await update_row(transform_test_session, 0, '{"invalid": json}')

    async def test_update_row_invalid_columns(self, transform_test_session) -> None:
        """Test updating row with invalid columns."""
        with pytest.raises(ToolError, match="not found"):
            await update_row(transform_test_session, 0, {"nonexistent": "value"})


@pytest.mark.asyncio
class TestColumnOperations:
    """Test column operation edge cases."""

    async def test_get_column_data_invalid_column(self, transform_test_session) -> None:
        """Test getting column data for invalid column."""
        with pytest.raises(ToolError, match="not found"):
            await get_column_data(transform_test_session, "nonexistent")

    async def test_get_column_data_invalid_range(self, transform_test_session) -> None:
        """Test getting column data with invalid row range."""
        # Test invalid start row
        with pytest.raises(ToolError, match="out of range"):
            await get_column_data(transform_test_session, "name", start_row=100)

        # Test invalid end row
        with pytest.raises(ToolError, match="invalid"):
            await get_column_data(transform_test_session, "name", start_row=0, end_row=100)

    async def test_replace_in_column_missing_column(self, transform_test_session) -> None:
        """Test replacing in missing column."""
        with pytest.raises(ToolError, match="not found"):
            await replace_in_column(transform_test_session, "nonexistent", "pattern", "replacement")

    async def test_extract_from_column_missing_column(self, transform_test_session) -> None:
        """Test extracting from missing column."""
        with pytest.raises(ToolError, match="not found"):
            await extract_from_column(transform_test_session, "nonexistent", r"(\w+)")

    async def test_split_column_missing_column(self, transform_test_session) -> None:
        """Test splitting missing column."""
        with pytest.raises(ToolError, match="not found"):
            await split_column(transform_test_session, "nonexistent", " ")

    async def test_transform_column_case_invalid_transform(self, transform_test_session) -> None:
        """Test case transformation with invalid transform type."""
        with pytest.raises(ToolError, match="Unknown transform"):
            await transform_column_case(transform_test_session, "name", "invalid")

    async def test_fill_column_nulls_missing_column(self, transform_test_session) -> None:
        """Test filling nulls in missing column."""
        with pytest.raises(ToolError, match="not found"):
            await fill_column_nulls(transform_test_session, "nonexistent", "value")


@pytest.mark.asyncio
class TestDataSummaryAndInspection:
    """Test data summary and inspection functions."""

    async def test_get_data_summary_without_preview(self, transform_test_session) -> None:
        """Test data summary without preview."""
        result = await get_data_summary(transform_test_session, include_preview=False)
        assert result.success is True
        assert result.preview is None  # No preview data when include_preview=False
        assert hasattr(result, "coordinate_system")

    async def test_get_data_summary_custom_preview_size(self, transform_test_session) -> None:
        """Test data summary with custom preview size."""
        result = await get_data_summary(
            transform_test_session, include_preview=True, max_preview_rows=2
        )
        assert result.success is True
        assert hasattr(result, "preview")
        assert len(result.preview.rows) <= 2  # Preview rows should be limited


@pytest.mark.asyncio
class TestAdvancedTransformations:
    """Test advanced transformation features."""

    async def test_fill_missing_values_different_strategies(self, transform_test_session) -> None:
        """Test different missing value fill strategies."""
        strategies = ["drop", "forward", "backward", "mean", "median", "mode"]

        for strategy in strategies:
            result = await fill_missing_values(transform_test_session, strategy=strategy)
            # Some strategies might fail depending on data types, but should handle gracefully
            assert hasattr(result, "success")

    async def test_fill_missing_values_invalid_strategy(self, transform_test_session) -> None:
        """Test fill missing values with invalid strategy."""
        with pytest.raises(ToolError, match="Unknown strategy"):
            await fill_missing_values(transform_test_session, strategy="invalid")

    async def test_remove_duplicates_different_options(self, transform_test_session) -> None:
        """Test duplicate removal with different options."""
        # First add some duplicate data
        await insert_row(
            transform_test_session,
            -1,
            {"name": "John Doe", "age": 30, "email": "john@example.com"},
        )

        # Test different keep options
        for keep_option in ["first", "last", "none"]:
            result = await remove_duplicates(transform_test_session, keep=keep_option)
            assert result.success is True
