"""Unit tests for transformation service functions."""

import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

from src import databeak
from src.databeak.exceptions import NoDataLoadedError, SessionNotFoundError
from src.databeak.services.transformation_service import (
    DuplicateRemovalResult,
    FillMissingResult,
    FilterResult,
    SortResult,
    StringOperationResult,
    _get_session_data,
    fill_missing_values_with_pydantic,
    filter_rows_with_pydantic,
    remove_duplicates_with_pydantic,
    sort_data_with_pydantic,
    strip_column_with_pydantic,
    transform_column_case_with_pydantic,
)


class TestGetSessionData:
    """Tests for _get_session_data helper function."""

    def test_get_session_data_success(self):
        """Test successful session and data retrieval."""
        # Create real session with test data
        session_id = str(uuid.uuid4())
        manager = databeak.session_manager
        session = manager.get_or_create_session(session_id)
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        session.df = df

        result_session, result_df = _get_session_data(session_id)

        assert result_session == session
        pd.testing.assert_frame_equal(result_df, df)

    def test_get_session_data_no_data(self):
        """Test exception when session has no data."""
        # Create a real session with no data (df = None)
        session_id = str(uuid.uuid4())
        manager = databeak.session_manager
        manager.get_or_create_session(session_id)
        # Don't set session.df, leaving it as None

        with pytest.raises(NoDataLoadedError):
            _get_session_data(session_id)

    def test_get_session_data_none_dataframe(self):
        """Test exception when DataFrame is None."""
        # Create a real session and explicitly set df to None
        session_id = str(uuid.uuid4())
        manager = databeak.session_manager
        session = manager.get_or_create_session(session_id)
        session.df = None

        with pytest.raises(NoDataLoadedError):
            _get_session_data(session_id)


class TestFilterRows:
    """Tests for filter_rows_with_pydantic function."""

    @pytest.fixture
    def session_with_data(self):
        """Fixture providing a real session with test data."""
        session_id = str(uuid.uuid4())
        manager = databeak.session_manager
        session = manager.get_or_create_session(session_id)
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "Diana"],
                "age": [25, 30, 35, 28],
                "city": ["NYC", "LA", "Chicago", "NYC"],
                "score": [85.5, 90.0, 78.5, 92.0],
            },
        )
        session.df = df.copy()
        return session_id, df

    @pytest.mark.asyncio
    async def test_filter_rows_equals_condition(self, session_with_data):
        """Test filtering with equals condition."""
        session_id, _original_df = session_with_data

        conditions = [{"column": "city", "operator": "==", "value": "NYC"}]
        result = await filter_rows_with_pydantic(session_id, conditions)

        assert isinstance(result, FilterResult)
        assert result.session_id == session_id
        assert result.rows_before == 4
        assert result.rows_after == 2
        assert result.rows_filtered == 2
        assert result.conditions_applied == 1

    @pytest.mark.asyncio
    async def test_filter_rows_greater_than(self, session_with_data):
        """Test filtering with greater than condition."""
        session_id, _original_df = session_with_data

        conditions = [{"column": "age", "operator": ">", "value": 30}]
        result = await filter_rows_with_pydantic(session_id, conditions)

        assert result.rows_filtered == 3  # 3 rows removed (Alice, Bob, Diana), 1 kept (Charlie)

    @pytest.mark.asyncio
    async def test_filter_rows_contains(self, session_with_data):
        """Test filtering with contains condition."""
        session_id, _original_df = session_with_data

        conditions = [{"column": "name", "operator": "contains", "value": "i"}]
        result = await filter_rows_with_pydantic(session_id, conditions)

        assert (
            result.rows_filtered == 1
        )  # Only Alice, Charlie, Diana contain 'i', so Bob is filtered out

    @pytest.mark.asyncio
    async def test_filter_rows_in_condition(self, session_with_data):
        """Test filtering with 'in' condition."""
        session_id, _original_df = session_with_data

        conditions = [{"column": "city", "operator": "in", "value": ["NYC", "LA"]}]
        result = await filter_rows_with_pydantic(session_id, conditions)

        assert result.rows_filtered == 1  # Removes Chicago

    @pytest.mark.asyncio
    async def test_filter_rows_or_mode(self, session_with_data):
        """Test filtering with OR mode."""
        session_id, _original_df = session_with_data

        conditions = [
            {"column": "age", "operator": "<", "value": 27},
            {"column": "city", "operator": "==", "value": "Chicago"},
        ]
        result = await filter_rows_with_pydantic(session_id, conditions, mode="or")

        assert result.rows_after >= 2  # Alice (25) or Charlie (Chicago)

    @pytest.mark.asyncio
    async def test_filter_rows_column_not_found(self, session_with_data):
        """Test filtering with non-existent column."""
        session_id, _original_df = session_with_data

        conditions = [{"column": "nonexistent", "operator": "==", "value": "test"}]

        with pytest.raises(ToolError, match="not found"):
            await filter_rows_with_pydantic(session_id, conditions)

    @pytest.mark.asyncio
    async def test_filter_rows_invalid_operator(self, session_with_data):
        """Test filtering with invalid operator."""
        session_id, _original_df = session_with_data

        conditions = [{"column": "age", "operator": "invalid_op", "value": 30}]

        with pytest.raises(ToolError, match="Invalid value for parameter"):
            await filter_rows_with_pydantic(session_id, conditions)

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_filter_rows_is_null(self, mock_get_session):
        """Test filtering with is_null condition."""
        session = MagicMock()
        df = pd.DataFrame({"name": ["Alice", None, "Charlie"], "age": [25, 30, None]})
        session.df = df.copy()
        mock_get_session.return_value = (session, df)

        conditions = [{"column": "name", "operator": "is_null"}]
        result = await filter_rows_with_pydantic("test-session", conditions)

        assert result.rows_after == 1  # Only the row with None name

    @pytest.mark.asyncio
    async def test_filter_rows_session_not_found_error(self):
        """Test handling of SessionNotFoundError."""
        with (
            patch(
                "src.databeak.services.transformation_service._get_session_data",
                side_effect=SessionNotFoundError("test-session"),
            ),
            pytest.raises(ToolError),
        ):
            await filter_rows_with_pydantic("invalid-session", [])


class TestSortData:
    """Tests for sort_data_with_pydantic function."""

    @pytest.fixture
    def mock_session_with_data(self):
        """Fixture providing a mock session with test data."""
        session = MagicMock()
        df = pd.DataFrame(
            {"name": ["Charlie", "Alice", "Bob"], "age": [35, 25, 30], "score": [78.5, 85.5, 90.0]},
        )
        session.df = df.copy()
        return session, df

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_sort_single_column(self, mock_get_session, mock_session_with_data):
        """Test sorting by single column."""
        session, original_df = mock_session_with_data
        mock_get_session.return_value = (session, original_df)

        result = await sort_data_with_pydantic("test-session", ["name"])

        assert isinstance(result, SortResult)
        assert result.session_id == "test-session"
        assert result.sorted_by == ["name"]
        assert result.ascending == [True]
        assert result.rows_affected == 3

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_sort_multiple_columns(self, mock_get_session, mock_session_with_data):
        """Test sorting by multiple columns."""
        session, original_df = mock_session_with_data
        mock_get_session.return_value = (session, original_df)

        result = await sort_data_with_pydantic(
            "test-session", ["age", "name"], ascending=[False, True]
        )

        assert result.sorted_by == ["age", "name"]
        assert result.ascending == [False, True]

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_sort_with_dict_format(self, mock_get_session, mock_session_with_data):
        """Test sorting with dictionary format columns."""
        session, original_df = mock_session_with_data
        mock_get_session.return_value = (session, original_df)

        columns = [{"column": "age", "ascending": False}, {"column": "name", "ascending": True}]
        result = await sort_data_with_pydantic("test-session", columns)

        assert result.sorted_by == ["age", "name"]
        assert result.ascending == [False, True]

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_sort_missing_column(self, mock_get_session, mock_session_with_data):
        """Test sorting with non-existent column."""
        session, original_df = mock_session_with_data
        mock_get_session.return_value = (session, original_df)

        with pytest.raises(ToolError, match="Columns not found"):
            await sort_data_with_pydantic("test-session", ["nonexistent"])


class TestRemoveDuplicates:
    """Tests for remove_duplicates_with_pydantic function."""

    @pytest.fixture
    def mock_session_with_duplicates(self):
        """Fixture providing a mock session with duplicate data."""
        session = MagicMock()
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Alice", "Charlie"],
                "age": [25, 30, 25, 35],
                "city": ["NYC", "LA", "NYC", "Chicago"],
            },
        )
        session.df = df.copy()
        return session, df

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_remove_duplicates_all_columns(
        self,
        mock_get_session,
        mock_session_with_duplicates,
    ):
        """Test removing duplicates considering all columns."""
        session, original_df = mock_session_with_duplicates
        mock_get_session.return_value = (session, original_df)

        result = await remove_duplicates_with_pydantic("test-session")

        assert isinstance(result, DuplicateRemovalResult)
        assert result.session_id == "test-session"
        assert result.rows_before == 4
        assert result.rows_after == 3  # One exact duplicate removed
        assert result.duplicates_removed == 1
        assert result.subset_columns is None
        assert result.keep_strategy == "first"

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_remove_duplicates_subset(self, mock_get_session, mock_session_with_duplicates):
        """Test removing duplicates based on subset of columns."""
        session, original_df = mock_session_with_duplicates
        mock_get_session.return_value = (session, original_df)

        result = await remove_duplicates_with_pydantic("test-session", subset=["name"])

        assert result.duplicates_removed == 1  # Alice appears twice
        assert result.subset_columns == ["name"]

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_remove_duplicates_keep_last(
        self,
        mock_get_session,
        mock_session_with_duplicates,
    ):
        """Test removing duplicates keeping last occurrence."""
        session, original_df = mock_session_with_duplicates
        mock_get_session.return_value = (session, original_df)

        result = await remove_duplicates_with_pydantic("test-session", keep="last")

        assert result.keep_strategy == "last"

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_remove_duplicates_subset_not_found(
        self,
        mock_get_session,
        mock_session_with_duplicates,
    ):
        """Test removing duplicates with non-existent subset columns."""
        session, original_df = mock_session_with_duplicates
        mock_get_session.return_value = (session, original_df)

        with pytest.raises(ToolError, match="Columns not found"):
            await remove_duplicates_with_pydantic("test-session", subset=["nonexistent"])


class TestFillMissingValues:
    """Tests for fill_missing_values_with_pydantic function."""

    @pytest.fixture
    def mock_session_with_nulls(self):
        """Fixture providing a mock session with missing data."""
        session = MagicMock()
        df = pd.DataFrame(
            {
                "name": ["Alice", None, "Charlie", "Diana"],
                "age": [25, None, 35, 28],
                "score": [85.5, 90.0, None, 92.0],
            },
        )
        session.df = df.copy()
        return session, df

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_fill_missing_drop_strategy(self, mock_get_session, mock_session_with_nulls):
        """Test filling missing values with drop strategy."""
        session, original_df = mock_session_with_nulls
        mock_get_session.return_value = (session, original_df)

        result = await fill_missing_values_with_pydantic("test-session", strategy="drop")

        assert isinstance(result, FillMissingResult)
        assert result.session_id == "test-session"
        assert result.strategy == "drop"
        assert result.nulls_before > 0
        assert result.nulls_after == 0

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_fill_missing_fill_strategy(self, mock_get_session, mock_session_with_nulls):
        """Test filling missing values with fill strategy."""
        session, original_df = mock_session_with_nulls
        mock_get_session.return_value = (session, original_df)

        result = await fill_missing_values_with_pydantic(
            "test-session",
            strategy="fill",
            value="MISSING",
        )

        assert result.strategy == "fill"
        assert result.fill_value == "MISSING"
        assert result.nulls_after < result.nulls_before

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_fill_missing_mean_strategy(self, mock_get_session, mock_session_with_nulls):
        """Test filling missing values with mean strategy."""
        session, original_df = mock_session_with_nulls
        mock_get_session.return_value = (session, original_df)

        result = await fill_missing_values_with_pydantic(
            "test-session",
            columns=["age"],
            strategy="mean",
        )

        assert result.strategy == "mean"
        assert result.columns_affected == ["age"]

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_fill_missing_forward_strategy(self, mock_get_session, mock_session_with_nulls):
        """Test filling missing values with forward fill strategy."""
        session, original_df = mock_session_with_nulls
        mock_get_session.return_value = (session, original_df)

        result = await fill_missing_values_with_pydantic("test-session", strategy="forward")

        assert result.strategy == "forward"

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_fill_missing_no_value_for_fill(self, mock_get_session, mock_session_with_nulls):
        """Test error when no value provided for fill strategy."""
        session, original_df = mock_session_with_nulls
        mock_get_session.return_value = (session, original_df)

        with pytest.raises(ToolError, match="Value required"):
            await fill_missing_values_with_pydantic("test-session", strategy="fill")

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_fill_missing_invalid_strategy(self, mock_get_session, mock_session_with_nulls):
        """Test error with invalid strategy."""
        session, original_df = mock_session_with_nulls
        mock_get_session.return_value = (session, original_df)

        with pytest.raises(ToolError, match="Unknown strategy"):
            await fill_missing_values_with_pydantic("test-session", strategy="invalid")

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_fill_missing_columns_not_found(self, mock_get_session, mock_session_with_nulls):
        """Test error when specified columns not found."""
        session, original_df = mock_session_with_nulls
        mock_get_session.return_value = (session, original_df)

        with pytest.raises(ToolError, match="Columns not found"):
            await fill_missing_values_with_pydantic("test-session", columns=["nonexistent"])


class TestTransformColumnCase:
    """Tests for transform_column_case_with_pydantic function."""

    @pytest.fixture
    def mock_session_with_text(self):
        """Fixture providing a mock session with text data."""
        session = MagicMock()
        df = pd.DataFrame(
            {
                "name": ["alice smith", "BOB JONES", "Charlie Brown"],
                "city": ["new york", "LOS ANGELES", "Chicago"],
            },
        )
        session.df = df.copy()
        return session, df

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_transform_column_upper(self, mock_get_session, mock_session_with_text):
        """Test transforming column to uppercase."""
        session, original_df = mock_session_with_text
        mock_get_session.return_value = (session, original_df)

        result = await transform_column_case_with_pydantic("test-session", "name", "upper")

        assert isinstance(result, StringOperationResult)
        assert result.session_id == "test-session"
        assert result.column == "name"
        assert result.operation == "case_upper"
        assert result.rows_affected == 3
        assert result.sample_before is not None
        assert result.sample_after is not None

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_transform_column_lower(self, mock_get_session, mock_session_with_text):
        """Test transforming column to lowercase."""
        session, original_df = mock_session_with_text
        mock_get_session.return_value = (session, original_df)

        result = await transform_column_case_with_pydantic("test-session", "name", "lower")

        assert result.operation == "case_lower"

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_transform_column_title(self, mock_get_session, mock_session_with_text):
        """Test transforming column to title case."""
        session, original_df = mock_session_with_text
        mock_get_session.return_value = (session, original_df)

        result = await transform_column_case_with_pydantic("test-session", "name", "title")

        assert result.operation == "case_title"

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_transform_column_capitalize(self, mock_get_session, mock_session_with_text):
        """Test transforming column to capitalize."""
        session, original_df = mock_session_with_text
        mock_get_session.return_value = (session, original_df)

        result = await transform_column_case_with_pydantic("test-session", "name", "capitalize")

        assert result.operation == "case_capitalize"

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_transform_column_not_found(self, mock_get_session, mock_session_with_text):
        """Test error when column not found."""
        session, original_df = mock_session_with_text
        mock_get_session.return_value = (session, original_df)

        with pytest.raises(ToolError, match="not found"):
            await transform_column_case_with_pydantic("test-session", "nonexistent", "upper")

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_transform_column_invalid_transform(
        self,
        mock_get_session,
        mock_session_with_text,
    ):
        """Test error with invalid transform."""
        session, original_df = mock_session_with_text
        mock_get_session.return_value = (session, original_df)

        with pytest.raises(ToolError, match="Unknown transform"):
            await transform_column_case_with_pydantic("test-session", "name", "invalid")


class TestStripColumn:
    """Tests for strip_column_with_pydantic function."""

    @pytest.fixture
    def mock_session_with_whitespace(self):
        """Fixture providing a mock session with whitespace data."""
        session = MagicMock()
        df = pd.DataFrame(
            {"name": ["  alice  ", " bob ", "charlie "], "city": [" NYC ", "LA", "  Chicago  "]},
        )
        session.df = df.copy()
        return session, df

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_strip_column(self, mock_get_session, mock_session_with_whitespace):
        """Test stripping whitespace from column."""
        session, original_df = mock_session_with_whitespace
        mock_get_session.return_value = (session, original_df)

        result = await strip_column_with_pydantic("test-session", "name")

        assert isinstance(result, StringOperationResult)
        assert result.session_id == "test-session"
        assert result.column == "name"
        assert result.operation == "strip"
        assert result.rows_affected == 3
        assert result.sample_before is not None
        assert result.sample_after is not None

    @pytest.mark.asyncio
    @patch("src.databeak.services.transformation_service._get_session_data")
    async def test_strip_column_not_found(self, mock_get_session, mock_session_with_whitespace):
        """Test error when column not found."""
        session, original_df = mock_session_with_whitespace
        mock_get_session.return_value = (session, original_df)

        with pytest.raises(ToolError, match="not found"):
            await strip_column_with_pydantic("test-session", "nonexistent")
