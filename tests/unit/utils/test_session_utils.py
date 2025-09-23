"""Unit tests for session_utils.py defensive programming helpers."""

import uuid
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.databeak.core.session import DatabeakSession
from src.databeak.exceptions import NoDataLoadedError
from src.databeak.utils.session_utils import (
    get_session_data,
    get_session_only,
    validate_session_has_data,
)


class TestGetSessionData:
    """Test get_session_data function for comprehensive validation."""

    def test_get_session_data_success(self):
        """Test successful session and data retrieval."""
        session_id = str(uuid.uuid4())

        with patch("src.databeak.utils.session_utils._session_manager") as mock_manager:
            # Mock session with data
            mock_session = Mock(spec=DatabeakSession)
            mock_session.has_data.return_value = True
            mock_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
            mock_session.df = mock_df

            mock_manager.get_or_create_session.return_value = mock_session

            session, df = get_session_data(session_id)

            assert session is mock_session
            assert df is mock_df
            mock_session.has_data.assert_called_once()

    def test_get_session_data_no_data_loaded(self):
        """Test get_session_data when session has no data."""
        session_id = str(uuid.uuid4())

        with patch("src.databeak.utils.session_utils._session_manager") as mock_manager:
            # Mock session without data
            mock_session = Mock(spec=DatabeakSession)
            mock_session.has_data.return_value = False

            mock_manager.get_or_create_session.return_value = mock_session

            with pytest.raises(NoDataLoadedError) as exc_info:
                get_session_data(session_id)

            assert str(exc_info.value) == f"No data loaded in session '{session_id}'"
            mock_session.has_data.assert_called_once()

    def test_get_session_data_df_is_none(self):
        """Test get_session_data when df is None despite has_data returning True."""
        session_id = str(uuid.uuid4())

        with patch("src.databeak.utils.session_utils._session_manager") as mock_manager:
            # Mock session with has_data=True but df=None (edge case)
            mock_session = Mock(spec=DatabeakSession)
            mock_session.has_data.return_value = True
            mock_session.df = None

            mock_manager.get_or_create_session.return_value = mock_session

            with pytest.raises(NoDataLoadedError) as exc_info:
                get_session_data(session_id)

            assert str(exc_info.value) == f"No data loaded in session '{session_id}'"

    def test_get_session_data_with_real_session(self):
        """Test get_session_data with actual DatabeakSession object."""
        session_id = str(uuid.uuid4())

        # Create real session and load data
        session = DatabeakSession(session_id)
        df = pd.DataFrame({"test": [1, 2, 3]})
        session.load_data(df)

        with patch("src.databeak.utils.session_utils._session_manager") as mock_manager:
            mock_manager.get_or_create_session.return_value = session

            result_session, result_df = get_session_data(session_id)

            assert result_session is session
            pd.testing.assert_frame_equal(result_df, df)


class TestGetSessionOnly:
    """Test get_session_only function for session retrieval without data requirement."""

    def test_get_session_only_success(self):
        """Test successful session retrieval without data requirement."""
        session_id = str(uuid.uuid4())

        with patch("src.databeak.utils.session_utils._session_manager") as mock_manager:
            mock_session = Mock(spec=DatabeakSession)
            mock_manager.get_or_create_session.return_value = mock_session

            session = get_session_only(session_id)

            assert session is mock_session
            # Should not call has_data() since data is not required
            mock_session.has_data.assert_not_called()

    def test_get_session_only_with_real_session(self):
        """Test get_session_only with actual DatabeakSession object."""
        session_id = str(uuid.uuid4())

        # Create real session (without loading data)
        real_session = DatabeakSession(session_id)

        with patch("src.databeak.utils.session_utils._session_manager") as mock_manager:
            mock_manager.get_or_create_session.return_value = real_session

            session = get_session_only(session_id)

            assert session is real_session
            assert session.session_id == session_id


class TestValidateSessionHasData:
    """Test validate_session_has_data function for existing session validation."""

    def test_validate_session_has_data_success(self):
        """Test successful validation of session with data."""
        session_id = str(uuid.uuid4())
        session = Mock(spec=DatabeakSession)
        session.has_data.return_value = True
        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        session.df = mock_df

        df = validate_session_has_data(session, session_id)

        assert df is mock_df
        session.has_data.assert_called_once()

    def test_validate_session_has_data_no_data(self):
        """Test validation failure when session has no data."""
        session_id = str(uuid.uuid4())
        session = Mock(spec=DatabeakSession)
        session.has_data.return_value = False

        with pytest.raises(NoDataLoadedError) as exc_info:
            validate_session_has_data(session, session_id)

        assert str(exc_info.value) == f"No data loaded in session '{session_id}'"
        session.has_data.assert_called_once()

    def test_validate_session_has_data_df_is_none(self):
        """Test validation when has_data returns True but df is None."""
        session_id = str(uuid.uuid4())
        session = Mock(spec=DatabeakSession)
        session.has_data.return_value = True
        session.df = None

        with pytest.raises(NoDataLoadedError) as exc_info:
            validate_session_has_data(session, session_id)

        assert str(exc_info.value) == f"No data loaded in session '{session_id}'"

    def test_validate_session_has_data_with_real_session(self):
        """Test validate_session_has_data with actual DatabeakSession object."""
        session_id = str(uuid.uuid4())
        session = DatabeakSession(session_id)
        df = pd.DataFrame({"test": [10, 20, 30]})
        session.load_data(df)

        result_df = validate_session_has_data(session, session_id)

        pd.testing.assert_frame_equal(result_df, df)


class TestSessionUtilsIntegration:
    """Integration tests for session_utils functions."""

    def test_defensive_programming_workflow(self):
        """Test complete defensive programming workflow."""
        session_id = str(uuid.uuid4())

        # Step 1: Get session without requiring data
        with patch("src.databeak.utils.session_utils._session_manager") as mock_manager:
            mock_session = Mock(spec=DatabeakSession)
            mock_manager.get_or_create_session.return_value = mock_session

            # Initially no data
            mock_session.has_data.return_value = False
            session = get_session_only(session_id)
            assert session is mock_session

            # Step 2: Later, after data is loaded
            mock_session.has_data.return_value = True
            mock_df = pd.DataFrame({"data": [1, 2, 3]})
            mock_session.df = mock_df

            # Step 3: Get session and data together
            session, df = get_session_data(session_id)
            assert session is mock_session
            assert df is mock_df

    def test_error_specificity(self):
        """Test that helper functions provide specific error types."""
        session_id = str(uuid.uuid4())

        with patch("src.databeak.utils.session_utils._session_manager") as mock_manager:
            mock_session = Mock(spec=DatabeakSession)
            mock_session.has_data.return_value = False
            mock_manager.get_or_create_session.return_value = mock_session

            # Should get NoDataLoadedError, not generic exception
            with pytest.raises(NoDataLoadedError):
                get_session_data(session_id)

            # validate_session_has_data should also give NoDataLoadedError
            with pytest.raises(NoDataLoadedError):
                validate_session_has_data(mock_session, session_id)

    def test_type_safety_tuple_unpacking(self):
        """Test that tuple unpacking provides type safety."""
        session_id = str(uuid.uuid4())

        with patch("src.databeak.utils.session_utils._session_manager") as mock_manager:
            mock_session = Mock(spec=DatabeakSession)
            mock_session.has_data.return_value = True
            mock_df = pd.DataFrame({"col": [1, 2, 3]})
            mock_session.df = mock_df

            mock_manager.get_or_create_session.return_value = mock_session

            # Tuple unpacking ensures both values are available
            session, df = get_session_data(session_id)

            # Both should be non-None after successful call
            assert session is not None
            assert df is not None
            assert len(df) == 3


class TestSessionUtilsEdgeCases:
    """Test edge cases and error scenarios for session_utils."""

    def test_different_session_ids(self):
        """Test that functions work with various session ID formats."""
        test_session_ids = [
            "simple-id",
            "uuid-" + str(uuid.uuid4()),
            "123-456-789",
            "test_session_with_underscores",
        ]

        for session_id in test_session_ids:
            with patch("src.databeak.utils.session_utils._session_manager") as mock_manager:
                mock_session = Mock(spec=DatabeakSession)
                mock_session.has_data.return_value = True
                mock_session.df = pd.DataFrame({"data": [1]})

                mock_manager.get_or_create_session.return_value = mock_session

                # Should work with any valid session ID
                session, df = get_session_data(session_id)
                assert session is mock_session
                assert len(df) == 1

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        session_id = str(uuid.uuid4())

        with patch("src.databeak.utils.session_utils._session_manager") as mock_manager:
            mock_session = Mock(spec=DatabeakSession)
            mock_session.has_data.return_value = True
            # Empty DataFrame is still valid data
            empty_df = pd.DataFrame()
            mock_session.df = empty_df

            mock_manager.get_or_create_session.return_value = mock_session

            session, df = get_session_data(session_id)

            assert session is mock_session
            assert len(df) == 0  # Empty DataFrame should be returned successfully


class TestSessionUtilsRegressionPrevention:
    """Regression tests to ensure session_utils maintains expected behavior."""

    def test_maintains_session_manager_contract(self):
        """Test that helpers maintain session manager contract."""
        session_id = str(uuid.uuid4())

        with patch("src.databeak.utils.session_utils._session_manager") as mock_manager:
            mock_session = Mock(spec=DatabeakSession)
            mock_session.has_data.return_value = True
            mock_session.df = pd.DataFrame({"test": [1, 2, 3]})

            mock_manager.get_or_create_session.return_value = mock_session

            # Should call get_or_create_session with correct session_id
            get_session_data(session_id)

            mock_manager.get_or_create_session.assert_called_once_with(session_id)

    def test_error_propagation(self):
        """Test that underlying exceptions are properly propagated."""
        session_id = str(uuid.uuid4())

        with patch("src.databeak.utils.session_utils._session_manager") as mock_manager:
            # Simulate session manager throwing an exception
            mock_manager.get_or_create_session.side_effect = RuntimeError(
                "Session manager error"
            )

            # Should propagate the underlying exception
            with pytest.raises(RuntimeError, match="Session manager error"):
                get_session_data(session_id)

    def test_no_side_effects_on_session_manager(self):
        """Test that helpers don't modify session manager state."""
        session_id = str(uuid.uuid4())

        with patch("src.databeak.utils.session_utils._session_manager") as mock_manager:
            mock_session = Mock(spec=DatabeakSession)
            mock_session.has_data.return_value = True
            mock_session.df = pd.DataFrame({"data": [1, 2, 3]})

            mock_manager.get_or_create_session.return_value = mock_session

            # Call helper function
            get_session_data(session_id)

            # Should only read from session manager, not modify it
            mock_manager.get_or_create_session.assert_called_once_with(session_id)
            # No other method calls on session manager
            assert len(mock_manager.method_calls) == 1
