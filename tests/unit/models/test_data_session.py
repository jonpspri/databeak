"""Unit tests for data_session.py module."""

import pandas as pd

from src.databeak.models.data_session import DataSession


class TestDataSession:
    """Tests for DataSession class."""

    def test_data_session_initialization(self):
        """Test DataSession initialization."""
        session = DataSession()
        assert session.df is None
        assert session.original_df is None
        assert session.file_path is None
        assert session.auto_save_enabled is True

    def test_has_data(self):
        """Test has_data method."""
        session = DataSession()
        assert session.has_data() is False

        session.df = pd.DataFrame({"col1": [1, 2, 3]})
        assert session.has_data() is True

    def test_is_modified(self):
        """Test is_modified property."""
        session = DataSession()
        session.df = pd.DataFrame({"col1": [1, 2, 3]})
        session.original_df = session.df.copy()
        assert session.is_modified is False

        session.df.loc[0, "col1"] = 999
        assert session.is_modified is True
