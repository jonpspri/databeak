"""Unit tests for data_session.py module."""

import pandas as pd

from src.databeak.models.data_session import DataSession


class TestDataSession:
    """Tests for DataSession class."""

    def test_data_session_initialization(self):
        """Test DataSession initialization."""
        session = DataSession(session_id="test-session-123")
        assert session.df is None
        assert session.original_df is None
        assert session.file_path is None
        assert session.session_id == "test-session-123"
        assert session.metadata == {}

    def test_has_data(self):
        """Test has_data method."""
        session = DataSession(session_id="test-session-456")
        assert session.has_data() is False

        session.df = pd.DataFrame({"col1": [1, 2, 3]})
        assert session.has_data() is True

    def test_load_data(self):
        """Test load_data method."""
        session = DataSession(session_id="test-session-789")
        df = pd.DataFrame({"col1": [1, 2, 3]})
        session.load_data(df, file_path="test.csv")

        assert session.df is not None
        assert session.original_df is not None
        assert session.file_path == "test.csv"
        assert len(session.df) == 3
        # Check that df and original_df are separate copies
        session.df.loc[0, "col1"] = 999
        assert session.original_df.loc[0, "col1"] == 1
