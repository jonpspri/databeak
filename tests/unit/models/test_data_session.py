"""Unit tests for data_session.py module."""

import pandas as pd
import pytest

from databeak.exceptions import NoDataLoadedError
from databeak.models.data_session import DataSession


class TestDataSession:
    """Tests for DataSession class."""

    def test_data_session_initialization(self) -> None:
        """Test DataSession initialization."""
        session = DataSession(session_id="test-session-123")
        assert session.df is None
        assert session.original_df is None
        assert session.file_path is None
        assert session.session_id == "test-session-123"
        assert session.metadata == {}

    def test_has_data(self) -> None:
        """Test has_data method."""
        session = DataSession(session_id="test-session-456")
        assert session.has_data() is False

        session.df = pd.DataFrame({"col1": [1, 2, 3]})
        assert session.has_data() is True

    def test_load_data(self) -> None:
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

    def test_validate_data_loaded_raises_when_no_data(self) -> None:
        """Test validate_data_loaded raises NoDataLoadedError when df is None."""
        session = DataSession(session_id="test-session-no-data")

        with pytest.raises(NoDataLoadedError) as exc_info:
            session.validate_data_loaded()

        assert "test-session-no-data" in str(exc_info.value)

    def test_validate_data_loaded_succeeds_when_data_present(self) -> None:
        """Test validate_data_loaded succeeds when data is loaded."""
        session = DataSession(session_id="test-session-with-data")
        session.df = pd.DataFrame({"col1": [1, 2, 3]})

        # Should not raise
        session.validate_data_loaded()

    def test_get_basic_stats_raises_when_no_data(self) -> None:
        """Test get_basic_stats raises NoDataLoadedError when df is None."""
        session = DataSession(session_id="test-session-stats-no-data")

        with pytest.raises(NoDataLoadedError) as exc_info:
            session.get_basic_stats()

        assert "test-session-stats-no-data" in str(exc_info.value)

    def test_get_basic_stats_succeeds_when_data_present(self) -> None:
        """Test get_basic_stats returns statistics when data is loaded."""
        session = DataSession(session_id="test-session-with-stats")
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, None, 6]})
        session.df = df

        stats = session.get_basic_stats()

        assert stats["row_count"] == 3
        assert stats["column_count"] == 2
        assert stats["null_counts"]["col1"] == 0
        assert stats["null_counts"]["col2"] == 1
        assert "memory_usage_mb" in stats

    def test_get_data_info_raises_when_no_data(self) -> None:
        """Test get_data_info raises NoDataLoadedError when df is None."""
        session = DataSession(session_id="test-session-info-no-data")

        with pytest.raises(NoDataLoadedError) as exc_info:
            session.get_data_info()

        assert "test-session-info-no-data" in str(exc_info.value)

    def test_clear_data(self) -> None:
        """Test clear_data removes all data and metadata."""
        session = DataSession(session_id="test-session-clear")
        df = pd.DataFrame({"col1": [1, 2, 3]})
        session.load_data(df, file_path="test.csv")

        assert session.has_data() is True
        assert session.metadata != {}

        session.clear_data()

        assert session.df is None
        assert session.original_df is None
        assert session.metadata == {}
        assert session.has_data() is False
