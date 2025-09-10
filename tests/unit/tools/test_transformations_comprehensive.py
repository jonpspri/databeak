"""Comprehensive unit tests for transformations module to achieve high coverage."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

from src.databeak.exceptions import (
    ColumnNotFoundError,
    InvalidParameterError,
    NoDataLoadedError,
    SessionNotFoundError,
)
from src.databeak.tools.transformations import (
    _get_session_data,
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


@pytest.fixture
def mock_session():
    """Create a mock session with test data."""
    session = Mock()
    session.data_session.df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 28, 32],
            "city": ["NYC", "LA", "Chicago", "NYC", "LA"],
            "salary": [50000, 60000, 70000, 55000, 65000],
            "email": ["alice@test.com", "bob@test.com", None, "diana@test.com", "eve@test.com"],
            "phone": ["123-456-7890", "234-567-8901", "345-678-9012", None, "456-789-0123"],
            "active": [True, False, True, True, False],
            "score": [85.5, 92.3, None, 88.7, 79.4],
        }
    )
    session.data_session.has_data.return_value = True
    session.data_session.record_operation = Mock()
    return session


@pytest.fixture
def mock_manager(mock_session):
    """Create a mock session manager."""
    with patch("src.databeak.tools.transformations.get_session_manager") as manager:
        manager.return_value.get_session.return_value = mock_session
        yield manager


class TestGetSessionData:
    """Test _get_session_data helper function."""

    def test_session_not_found(self):
        """Test when session doesn't exist."""
        with patch("src.databeak.tools.transformations.get_session_manager") as manager:
            manager.return_value.get_session.return_value = None
            with pytest.raises(SessionNotFoundError):
                _get_session_data("invalid-session")

    def test_no_data_loaded(self):
        """Test when session has no data."""
        with patch("src.databeak.tools.transformations.get_session_manager") as manager:
            session = Mock()
            session.data_session.has_data.return_value = False
            manager.return_value.get_session.return_value = session
            with pytest.raises(NoDataLoadedError):
                _get_session_data("empty-session")

    def test_valid_session(self, mock_manager, mock_session):
        """Test with valid session and data."""
        session, df = _get_session_data("test-session")
        assert session == mock_session
        assert df is not None
        assert len(df) == 5


@pytest.mark.asyncio
class TestFilterRows:
    """Test filter_rows function with all operators."""

    async def test_filter_equals(self, mock_manager):
        """Test == operator."""
        result = await filter_rows(
            "test-session", [{"column": "city", "operator": "==", "value": "NYC"}]
        )
        assert result.success is True
        assert result.rows_after == 2

    async def test_filter_not_equals(self, mock_manager):
        """Test != operator."""
        result = await filter_rows(
            "test-session", [{"column": "active", "operator": "!=", "value": True}]
        )
        assert result.success is True
        assert result.rows_after == 2

    async def test_filter_greater_than(self, mock_manager):
        """Test > operator."""
        result = await filter_rows(
            "test-session", [{"column": "age", "operator": ">", "value": 30}]
        )
        assert result.success is True
        assert result.rows_after == 2

    async def test_filter_less_than(self, mock_manager):
        """Test < operator."""
        result = await filter_rows(
            "test-session", [{"column": "salary", "operator": "<", "value": 60000}]
        )
        assert result.success is True
        assert result.rows_after == 2

    async def test_filter_greater_equal(self, mock_manager):
        """Test >= operator."""
        result = await filter_rows(
            "test-session", [{"column": "age", "operator": ">=", "value": 30}]
        )
        assert result.success is True
        assert result.rows_after == 3

    async def test_filter_less_equal(self, mock_manager):
        """Test <= operator."""
        result = await filter_rows(
            "test-session", [{"column": "age", "operator": "<=", "value": 30}]
        )
        assert result.success is True
        assert result.rows_after == 3

    async def test_filter_contains(self, mock_manager):
        """Test contains operator."""
        result = await filter_rows(
            "test-session", [{"column": "email", "operator": "contains", "value": "@test.com"}]
        )
        assert result.success is True
        assert result.rows_after == 4

    async def test_filter_starts_with(self, mock_manager):
        """Test starts_with operator."""
        result = await filter_rows(
            "test-session", [{"column": "name", "operator": "starts_with", "value": "D"}]
        )
        assert result.success is True
        assert result.rows_after == 1

    async def test_filter_ends_with(self, mock_manager):
        """Test ends_with operator."""
        result = await filter_rows(
            "test-session", [{"column": "phone", "operator": "ends_with", "value": "0123"}]
        )
        assert result.success is True
        assert result.rows_after == 1

    async def test_filter_in(self, mock_manager):
        """Test in operator."""
        result = await filter_rows(
            "test-session", [{"column": "city", "operator": "in", "value": ["NYC", "LA"]}]
        )
        assert result.success is True
        assert result.rows_after == 4

    async def test_filter_not_in(self, mock_manager):
        """Test not_in operator."""
        result = await filter_rows(
            "test-session", [{"column": "city", "operator": "not_in", "value": ["NYC", "LA"]}]
        )
        assert result.success is True
        assert result.rows_after == 1

    async def test_filter_is_null(self, mock_manager):
        """Test is_null operator."""
        result = await filter_rows("test-session", [{"column": "email", "operator": "is_null"}])
        assert result.success is True
        assert result.rows_after == 1

    async def test_filter_not_null(self, mock_manager):
        """Test not_null operator."""
        result = await filter_rows("test-session", [{"column": "phone", "operator": "not_null"}])
        assert result.success is True
        assert result.rows_after == 4

    async def test_filter_or_mode(self, mock_manager):
        """Test OR mode for combining conditions."""
        result = await filter_rows(
            "test-session",
            [
                {"column": "city", "operator": "==", "value": "NYC"},
                {"column": "age", "operator": ">", "value": 32},
            ],
            mode="or",
        )
        assert result.success is True
        assert result.rows_after == 3

    async def test_filter_and_mode(self, mock_manager):
        """Test AND mode for combining conditions."""
        result = await filter_rows(
            "test-session",
            [
                {"column": "city", "operator": "==", "value": "NYC"},
                {"column": "age", "operator": "<", "value": 30},
            ],
            mode="and",
        )
        assert result.success is True
        assert result.rows_after == 2

    async def test_filter_invalid_column(self, mock_manager):
        """Test filtering with invalid column."""
        with pytest.raises(ColumnNotFoundError):
            await filter_rows(
                "test-session", [{"column": "invalid_col", "operator": "==", "value": "test"}]
            )

    async def test_filter_invalid_operator(self, mock_manager):
        """Test filtering with invalid operator."""
        with pytest.raises(InvalidParameterError):
            await filter_rows(
                "test-session", [{"column": "name", "operator": "invalid", "value": "test"}]
            )


@pytest.mark.asyncio
class TestSortData:
    """Test sort_data function."""

    async def test_sort_ascending(self, mock_manager):
        """Test sorting in ascending order."""
        result = await sort_data("test-session", columns=["age"])
        assert result.success is True
        assert result.sorted_columns == ["age"]
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df.iloc[0]["age"] == 25

    async def test_sort_descending(self, mock_manager):
        """Test sorting in descending order."""
        result = await sort_data("test-session", columns=["salary"], ascending=False)
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df.iloc[0]["salary"] == 70000

    async def test_sort_multiple_columns(self, mock_manager):
        """Test sorting by multiple columns."""
        result = await sort_data("test-session", columns=["city", "age"], ascending=[True, False])
        assert result.success is True
        assert result.sorted_columns == ["city", "age"]

    async def test_sort_invalid_column(self, mock_manager):
        """Test sorting with invalid column."""
        with pytest.raises(ColumnNotFoundError):
            await sort_data("test-session", columns=["invalid_col"])


@pytest.mark.asyncio
class TestColumnOperations:
    """Test column manipulation functions."""

    async def test_add_column_with_value(self, mock_manager):
        """Test adding column with default value."""
        result = await add_column("test-session", "new_col", value="default")
        assert result.success is True
        assert result.column_name == "new_col"
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert "new_col" in df.columns
        assert all(df["new_col"] == "default")

    async def test_add_column_with_list(self, mock_manager):
        """Test adding column with list of values."""
        values = [1, 2, 3, 4, 5]
        result = await add_column("test-session", "numbers", value=values)
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert list(df["numbers"]) == values

    async def test_add_column_with_formula(self, mock_manager):
        """Test adding column with formula."""
        result = await add_column("test-session", "age_doubled", formula="age * 2")
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df["age_doubled"].iloc[0] == 50

    async def test_add_column_existing(self, mock_manager):
        """Test adding column that already exists."""
        with pytest.raises(InvalidParameterError):
            await add_column("test-session", "name", value="test")

    async def test_remove_columns(self, mock_manager):
        """Test removing columns."""
        result = await remove_columns("test-session", ["email", "phone"])
        assert result.success is True
        assert result.removed_columns == ["email", "phone"]
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert "email" not in df.columns
        assert "phone" not in df.columns

    async def test_remove_columns_invalid(self, mock_manager):
        """Test removing non-existent columns."""
        with pytest.raises(ColumnNotFoundError):
            await remove_columns("test-session", ["invalid_col"])

    async def test_rename_columns(self, mock_manager):
        """Test renaming columns."""
        result = await rename_columns("test-session", {"name": "full_name", "age": "years"})
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert "full_name" in df.columns
        assert "years" in df.columns
        assert "name" not in df.columns
        assert "age" not in df.columns

    async def test_select_columns(self, mock_manager):
        """Test selecting specific columns."""
        result = await select_columns("test-session", ["name", "age", "city"])
        assert result.success is True
        assert result.selected_columns == ["name", "age", "city"]
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert list(df.columns) == ["name", "age", "city"]

    async def test_change_column_type_to_string(self, mock_manager):
        """Test changing column type to string."""
        result = await change_column_type("test-session", "age", "str")
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df["age"].dtype == object

    async def test_change_column_type_to_int(self, mock_manager):
        """Test changing column type to int."""
        mock_manager.return_value.get_session.return_value.data_session.df["score"] = (
            mock_manager.return_value.get_session.return_value.data_session.df["score"].fillna(0)
        )
        result = await change_column_type("test-session", "score", "int")
        assert result.success is True

    async def test_change_column_type_to_float(self, mock_manager):
        """Test changing column type to float."""
        result = await change_column_type("test-session", "age", "float")
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df["age"].dtype == float

    async def test_change_column_type_to_bool(self, mock_manager):
        """Test changing column type to bool."""
        # Create a column with 0s and 1s
        df = mock_manager.return_value.get_session.return_value.data_session.df
        df["binary"] = [0, 1, 0, 1, 1]
        result = await change_column_type("test-session", "binary", "bool")
        assert result.success is True
        assert df["binary"].dtype == bool

    async def test_change_column_type_to_datetime(self, mock_manager):
        """Test changing column type to datetime."""
        df = mock_manager.return_value.get_session.return_value.data_session.df
        df["date"] = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        result = await change_column_type("test-session", "date", "datetime")
        assert result.success is True
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    async def test_change_column_type_invalid(self, mock_manager):
        """Test changing to invalid type."""
        with pytest.raises(InvalidParameterError):
            await change_column_type("test-session", "age", "invalid_type")


@pytest.mark.asyncio
class TestCellOperations:
    """Test cell-level operations."""

    async def test_get_cell_value(self, mock_manager):
        """Test getting cell value."""
        result = await get_cell_value("test-session", 0, "name")
        assert result.success is True
        assert result.value == "Alice"
        assert result.row_index == 0
        assert result.column == "name"

    async def test_get_cell_value_null(self, mock_manager):
        """Test getting null cell value."""
        result = await get_cell_value("test-session", 2, "email")
        assert result.success is True
        assert result.value is None
        assert result.is_null is True

    async def test_get_cell_value_invalid_row(self, mock_manager):
        """Test getting cell with invalid row index."""
        with pytest.raises(InvalidParameterError):
            await get_cell_value("test-session", 10, "name")

    async def test_get_cell_value_invalid_column(self, mock_manager):
        """Test getting cell with invalid column."""
        with pytest.raises(ColumnNotFoundError):
            await get_cell_value("test-session", 0, "invalid_col")

    async def test_set_cell_value(self, mock_manager):
        """Test setting cell value."""
        result = await set_cell_value("test-session", 0, "age", 26)
        assert result.success is True
        assert result.old_value == 25
        assert result.new_value == 26
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df.loc[0, "age"] == 26

    async def test_set_cell_value_null(self, mock_manager):
        """Test setting cell to null."""
        result = await set_cell_value("test-session", 0, "email", None)
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert pd.isna(df.loc[0, "email"])


@pytest.mark.asyncio
class TestRowOperations:
    """Test row-level operations."""

    async def test_get_row_data(self, mock_manager):
        """Test getting row data."""
        result = await get_row_data("test-session", 0)
        assert result.success is True
        assert result.row_index == 0
        assert result.data["name"] == "Alice"
        assert result.data["age"] == 25

    async def test_get_row_data_invalid_index(self, mock_manager):
        """Test getting row with invalid index."""
        with pytest.raises(InvalidParameterError):
            await get_row_data("test-session", 10)

    async def test_update_row(self, mock_manager):
        """Test updating row data."""
        new_data = {"name": "Alice Updated", "age": 26, "city": "Boston"}
        result = await update_row("test-session", 0, new_data)
        assert result.success is True
        assert result.row_index == 0
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df.loc[0, "name"] == "Alice Updated"
        assert df.loc[0, "age"] == 26
        assert df.loc[0, "city"] == "Boston"

    async def test_insert_row_at_position(self, mock_manager):
        """Test inserting row at specific position."""
        new_row = {"name": "Frank", "age": 40, "city": "Seattle"}
        result = await insert_row("test-session", new_row, position=2)
        assert result.success is True
        assert result.position == 2
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df.iloc[2]["name"] == "Frank"
        assert len(df) == 6

    async def test_insert_row_append(self, mock_manager):
        """Test appending row at the end."""
        new_row = {"name": "Grace", "age": 45, "city": "Miami"}
        result = await insert_row("test-session", new_row)
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df.iloc[-1]["name"] == "Grace"
        assert len(df) == 6

    async def test_delete_row(self, mock_manager):
        """Test deleting row."""
        result = await delete_row("test-session", 2)
        assert result.success is True
        assert result.row_index == 2
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert len(df) == 4
        assert "Charlie" not in df["name"].values

    async def test_delete_row_invalid_index(self, mock_manager):
        """Test deleting row with invalid index."""
        with pytest.raises(InvalidParameterError):
            await delete_row("test-session", 10)


@pytest.mark.asyncio
class TestColumnData:
    """Test column data operations."""

    async def test_get_column_data(self, mock_manager):
        """Test getting column data."""
        result = await get_column_data("test-session", "name")
        assert result.success is True
        assert result.column == "name"
        assert len(result.data) == 5
        assert result.data[0] == "Alice"
        assert result.null_count == 0

    async def test_get_column_data_with_nulls(self, mock_manager):
        """Test getting column data with nulls."""
        result = await get_column_data("test-session", "email")
        assert result.success is True
        assert result.null_count == 1
        assert None in result.data

    async def test_get_column_data_invalid(self, mock_manager):
        """Test getting invalid column data."""
        with pytest.raises(ColumnNotFoundError):
            await get_column_data("test-session", "invalid_col")


@pytest.mark.asyncio
class TestDataCleaning:
    """Test data cleaning operations."""

    async def test_remove_duplicates_all(self, mock_manager):
        """Test removing duplicates from all columns."""
        # Add a duplicate row
        df = mock_manager.return_value.get_session.return_value.data_session.df
        duplicate = df.iloc[0].copy()
        df.loc[len(df)] = duplicate

        result = await remove_duplicates("test-session")
        assert result.success is True
        assert result.removed_count == 1
        assert len(df) == 5

    async def test_remove_duplicates_subset(self, mock_manager):
        """Test removing duplicates based on subset of columns."""
        # Add a row with duplicate city
        df = mock_manager.return_value.get_session.return_value.data_session.df
        df.loc[len(df)] = [
            "New Person",
            50,
            "NYC",
            80000,
            "new@test.com",
            "999-999-9999",
            True,
            95.0,
        ]

        result = await remove_duplicates("test-session", subset=["city"])
        assert result.success is True
        assert result.removed_count > 0

    async def test_remove_duplicates_keep_last(self, mock_manager):
        """Test removing duplicates keeping last occurrence."""
        df = mock_manager.return_value.get_session.return_value.data_session.df
        duplicate = df.iloc[0].copy()
        df.loc[len(df)] = duplicate

        result = await remove_duplicates("test-session", keep="last")
        assert result.success is True
        assert result.removed_count == 1

    async def test_fill_missing_values_mean(self, mock_manager):
        """Test filling missing values with mean."""
        result = await fill_missing_values("test-session", columns=["score"], strategy="mean")
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df["score"].isna().sum() == 0

    async def test_fill_missing_values_median(self, mock_manager):
        """Test filling missing values with median."""
        result = await fill_missing_values("test-session", columns=["score"], strategy="median")
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df["score"].isna().sum() == 0

    async def test_fill_missing_values_mode(self, mock_manager):
        """Test filling missing values with mode."""
        result = await fill_missing_values("test-session", columns=["city"], strategy="mode")
        assert result.success is True

    async def test_fill_missing_values_constant(self, mock_manager):
        """Test filling missing values with constant."""
        result = await fill_missing_values(
            "test-session", columns=["email"], strategy="constant", value="unknown@test.com"
        )
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df["email"].isna().sum() == 0

    async def test_fill_missing_values_forward(self, mock_manager):
        """Test filling missing values with forward fill."""
        result = await fill_missing_values("test-session", columns=["phone"], strategy="forward")
        assert result.success is True

    async def test_fill_missing_values_backward(self, mock_manager):
        """Test filling missing values with backward fill."""
        result = await fill_missing_values("test-session", columns=["phone"], strategy="backward")
        assert result.success is True

    async def test_fill_column_nulls(self, mock_manager):
        """Test filling nulls in specific column."""
        result = await fill_column_nulls("test-session", "email", "no-email@test.com")
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df["email"].isna().sum() == 0


@pytest.mark.asyncio
class TestStringOperations:
    """Test string manipulation operations."""

    async def test_split_column(self, mock_manager):
        """Test splitting column by delimiter."""
        df = mock_manager.return_value.get_session.return_value.data_session.df
        df["full_address"] = [
            "123 Main St, NYC",
            "456 Oak Ave, LA",
            "789 Pine Rd, Chicago",
            "321 Elm St, NYC",
            "654 Maple Dr, LA",
        ]

        result = await split_column(
            "test-session", "full_address", ",", new_columns=["street", "city_state"]
        )
        assert result.success is True
        assert "street" in df.columns
        assert "city_state" in df.columns

    async def test_split_column_expand(self, mock_manager):
        """Test splitting column with expand."""
        df = mock_manager.return_value.get_session.return_value.data_session.df

        result = await split_column("test-session", "phone", "-", expand=True)
        assert result.success is True
        assert "phone_0" in df.columns
        assert "phone_1" in df.columns
        assert "phone_2" in df.columns

    async def test_strip_column(self, mock_manager):
        """Test stripping whitespace from column."""
        df = mock_manager.return_value.get_session.return_value.data_session.df
        df["name"] = ["  Alice  ", " Bob", "Charlie ", " Diana ", "Eve"]

        result = await strip_column("test-session", "name")
        assert result.success is True
        assert df["name"].iloc[0] == "Alice"
        assert df["name"].iloc[1] == "Bob"

    async def test_extract_from_column(self, mock_manager):
        """Test extracting pattern from column."""
        result = await extract_from_column("test-session", "phone", r"(\d{3})", "area_code")
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert "area_code" in df.columns

    async def test_transform_column_case_upper(self, mock_manager):
        """Test transforming column to uppercase."""
        result = await transform_column_case("test-session", "name", "upper")
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df["name"].iloc[0] == "ALICE"

    async def test_transform_column_case_lower(self, mock_manager):
        """Test transforming column to lowercase."""
        result = await transform_column_case("test-session", "name", "lower")
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df["name"].iloc[0] == "alice"

    async def test_transform_column_case_title(self, mock_manager):
        """Test transforming column to title case."""
        df = mock_manager.return_value.get_session.return_value.data_session.df
        df["name"] = ["john doe", "jane smith", "bob johnson", "alice brown", "eve davis"]

        result = await transform_column_case("test-session", "name", "title")
        assert result.success is True
        assert df["name"].iloc[0] == "John Doe"

    async def test_replace_in_column(self, mock_manager):
        """Test replacing values in column."""
        result = await replace_in_column("test-session", "city", "NYC", "New York City")
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert "New York City" in df["city"].values
        assert "NYC" not in df["city"].values

    async def test_replace_in_column_regex(self, mock_manager):
        """Test replacing with regex in column."""
        result = await replace_in_column(
            "test-session", "phone", r"(\d{3})-(\d{3})-(\d{4})", r"(\1) \2-\3", regex=True
        )
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        # Check that phone numbers are reformatted
        assert "(" in str(df["phone"].iloc[0])


@pytest.mark.asyncio
class TestUpdateColumn:
    """Test update_column function."""

    async def test_update_column_with_value(self, mock_manager):
        """Test updating column with constant value."""
        result = await update_column("test-session", "city", value="Updated City")
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert all(df["city"] == "Updated City")

    async def test_update_column_with_formula(self, mock_manager):
        """Test updating column with formula."""
        result = await update_column("test-session", "salary", formula="salary * 1.1")
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert df["salary"].iloc[0] == 55000  # 50000 * 1.1

    async def test_update_column_with_mapping(self, mock_manager):
        """Test updating column with value mapping."""
        result = await update_column(
            "test-session", "city", mapping={"NYC": "New York", "LA": "Los Angeles"}
        )
        assert result.success is True
        df = mock_manager.return_value.get_session.return_value.data_session.df
        assert "New York" in df["city"].values
        assert "Los Angeles" in df["city"].values

    async def test_update_column_invalid(self, mock_manager):
        """Test updating non-existent column."""
        with pytest.raises(ColumnNotFoundError):
            await update_column("test-session", "invalid_col", value="test")


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling in various scenarios."""

    async def test_invalid_session(self):
        """Test operations with invalid session."""
        with patch("src.databeak.tools.transformations.get_session_manager") as manager:
            manager.return_value.get_session.return_value = None

            with pytest.raises(SessionNotFoundError):
                await filter_rows("invalid", [])

    async def test_no_data_in_session(self):
        """Test operations with no data loaded."""
        with patch("src.databeak.tools.transformations.get_session_manager") as manager:
            session = Mock()
            session.data_session.has_data.return_value = False
            manager.return_value.get_session.return_value = session

            with pytest.raises(NoDataLoadedError):
                await filter_rows("empty", [])

    async def test_invalid_mode_filter(self, mock_manager):
        """Test filter with invalid mode."""
        with pytest.raises(InvalidParameterError):
            await filter_rows(
                "test-session",
                [{"column": "age", "operator": "==", "value": 30}],
                mode="invalid_mode",
            )

    async def test_formula_evaluation_error(self, mock_manager):
        """Test formula with evaluation error."""
        with pytest.raises(ToolError):
            await add_column("test-session", "bad_formula", formula="nonexistent_column * 2")

    async def test_list_length_mismatch(self, mock_manager):
        """Test adding column with wrong list length."""
        with pytest.raises(InvalidParameterError):
            await add_column(
                "test-session",
                "wrong_length",
                value=[1, 2, 3],  # Only 3 values for 5 rows
            )

    async def test_invalid_dtype_conversion(self, mock_manager):
        """Test invalid dtype conversion."""
        df = mock_manager.return_value.get_session.return_value.data_session.df
        df["text"] = ["abc", "def", "ghi", "jkl", "mno"]

        with pytest.raises(ToolError):
            await change_column_type("test-session", "text", "int")

    async def test_invalid_strategy_fill(self, mock_manager):
        """Test fill missing with invalid strategy."""
        with pytest.raises(InvalidParameterError):
            await fill_missing_values(
                "test-session", columns=["score"], strategy="invalid_strategy"
            )

    async def test_invalid_case_transform(self, mock_manager):
        """Test transform case with invalid option."""
        with pytest.raises(InvalidParameterError):
            await transform_column_case("test-session", "name", "invalid_case")
