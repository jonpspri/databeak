"""Unit tests for session.py module."""

import uuid
from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from databeak.core.session import (
    DatabeakSession,
    SessionManager,
    get_session_manager,
)
from databeak.core.settings import DataBeakSettings
from databeak.models.data_models import ExportFormat


class TestDataBeakSettings:
    """Tests for DataBeakSettings configuration."""

    def test_default_settings(self):
        """Test default settings initialization."""
        settings = DataBeakSettings()
        assert settings.session_timeout == 3600
        # csv_history_dir removed - history functionality eliminated
        assert settings.max_file_size_mb == 1024
        assert settings.memory_threshold_mb == 2048
        assert settings.max_anomaly_sample_size == 10000  # Anomaly detection sample size


class TestDatabeakSession:
    """Tests for DatabeakSession class functionality."""

    def test_df_property_setter_and_getter(self):
        """Test DataFrame property setter and getter."""
        session = DatabeakSession()
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        # Test setter
        session.df = df
        assert session.df is not None
        assert len(session.df) == 2

        # Test getter
        retrieved_df = session.df
        pd.testing.assert_frame_equal(retrieved_df, df)

    def test_df_property_deleter(self):
        """Test DataFrame property deleter (lines 109-113)."""
        session = DatabeakSession()
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        session.df = df

        # Verify data is there
        assert session.df is not None

        # Test deleter
        del session.df
        assert session.df is None

    def test_has_data_method(self):
        """Test has_data method (line 117)."""
        session = DatabeakSession()

        # Initially no data
        assert not session.has_data()

        # Load data
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        session.df = df
        assert session.has_data()

        # Clear data
        del session.df
        assert not session.has_data()

    @pytest.mark.asyncio
    async def test_save_callback_csv_format(self, tmp_path):
        """Test _save_callback with CSV format (lines 199-227)."""
        session = DatabeakSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        file_path = str(tmp_path / "test.csv")
        result = await session._save_callback(file_path, ExportFormat.CSV, "utf-8")

        assert result["success"] is True
        assert result["file_path"] == file_path
        assert result["rows"] == 2
        assert result["columns"] == 2
        assert Path(file_path).exists()

    @pytest.mark.asyncio
    async def test_save_callback_tsv_format(self, tmp_path):
        """Test _save_callback with TSV format."""
        session = DatabeakSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        file_path = str(tmp_path / "test.tsv")
        result = await session._save_callback(file_path, ExportFormat.TSV, "utf-8")

        assert result["success"] is True
        assert Path(file_path).exists()

        # Verify TSV format (tab-separated)
        content = Path(file_path).read_text()
        assert "\t" in content

    @pytest.mark.asyncio
    async def test_save_callback_json_format(self, tmp_path):
        """Test _save_callback with JSON format."""
        session = DatabeakSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        file_path = str(tmp_path / "test.json")
        result = await session._save_callback(file_path, ExportFormat.JSON, "utf-8")

        assert result["success"] is True
        assert Path(file_path).exists()

    @pytest.mark.asyncio
    async def test_save_callback_excel_format(self, tmp_path):
        """Test _save_callback with Excel format."""
        session = DatabeakSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        file_path = str(tmp_path / "test.xlsx")
        result = await session._save_callback(file_path, ExportFormat.EXCEL, "utf-8")

        assert result["success"] is True
        assert Path(file_path).exists()

    @pytest.mark.asyncio
    async def test_save_callback_parquet_format(self, tmp_path):
        """Test _save_callback with Parquet format."""
        session = DatabeakSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        file_path = str(tmp_path / "test.parquet")
        result = await session._save_callback(file_path, ExportFormat.PARQUET, "utf-8")

        assert result["success"] is True
        assert Path(file_path).exists()

    @pytest.mark.asyncio
    async def test_save_callback_unsupported_format(self, tmp_path):
        """Test _save_callback with unsupported format."""
        session = DatabeakSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        file_path = str(tmp_path / "test.unknown")
        # Use a string that's not in ExportFormat enum
        result = await session._save_callback(file_path, "UNKNOWN", "utf-8")

        assert result["success"] is False
        assert "Unsupported format" in result["error"]

    @pytest.mark.asyncio
    async def test_save_callback_no_data(self, tmp_path):
        """Test _save_callback when no data is loaded."""
        session = DatabeakSession()
        # Don't load any data

        file_path = str(tmp_path / "test.csv")
        result = await session._save_callback(file_path, ExportFormat.CSV, "utf-8")

        assert result["success"] is False
        assert "No data to save" in result["error"]

    @pytest.mark.asyncio
    async def test_save_callback_exception_handling(self, tmp_path):
        """Test _save_callback exception handling."""
        session = DatabeakSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        # Use invalid path to trigger exception
        invalid_path = "/invalid/path/that/does/not/exist/test.csv"
        result = await session._save_callback(invalid_path, ExportFormat.CSV, "utf-8")

        assert result["success"] is False
        assert "error" in result


class TestSessionManager:
    """Tests for SessionManager functionality."""

    def test_get_session_manager(self):
        """Test getting session manager instance."""
        manager = get_session_manager()
        assert manager is not None
        # Singleton pattern
        manager2 = get_session_manager()
        assert manager is manager2

    def test_session_manager_init(self):
        """Test SessionManager initialization."""
        manager = SessionManager(max_sessions=10, ttl_minutes=30)
        assert manager.max_sessions == 10
        assert manager.ttl_minutes == 30
        assert len(manager.sessions) == 0
        assert len(manager.sessions_to_cleanup) == 0

    def test_get_session_creates_new(self):
        """Test get_session creates new session when needed."""
        manager = SessionManager()
        session_id = str(uuid.uuid4())
        _session = manager.get_or_create_session(session_id)

        assert session_id is not None
        assert session_id in manager.sessions
        assert len(manager.sessions) == 1

    def test_get_session_max_sessions_limit(self):
        """Test get_session when max sessions limit is reached (lines 453-454)."""
        manager = SessionManager(max_sessions=2, ttl_minutes=60)

        # Create max sessions
        session1_id = str(uuid.uuid4())
        session2_id = str(uuid.uuid4())
        manager.get_or_create_session(session1_id)
        manager.get_or_create_session(session2_id)
        assert len(manager.sessions) == 2

        # Mock the oldest session to have older access time using datetime
        from datetime import datetime, timedelta

        old_time = datetime.now(UTC) - timedelta(hours=1)
        oldest_session = manager.sessions[session1_id]

        with patch.object(oldest_session.lifecycle, "last_accessed", old_time):
            # Create third session - should remove oldest
            session3_id = str(uuid.uuid4())
            manager.get_or_create_session(session3_id)

            assert len(manager.sessions) == 2
            assert session1_id not in manager.sessions  # Oldest removed
            assert session2_id in manager.sessions
            assert session3_id in manager.sessions

    def test_get_session_valid(self):
        """Test getting a valid, non-expired session."""
        manager = SessionManager()
        session_id = str(uuid.uuid4())
        _session = manager.get_or_create_session(session_id)

        retrieved_session = manager.get_or_create_session(session_id)
        assert retrieved_session is not None
        assert retrieved_session.session_id == session_id

    def test_get_session_nonexistent(self):
        """Test getting a non-existent session."""
        manager = SessionManager()
        # get_session creates if not exists, so check sessions dict directly
        retrieved_session = manager.sessions.get("nonexistent-id")
        assert retrieved_session is None

    @pytest.mark.skip(reason="Session expiration logic needs clarification")
    def test_get_session_expired(self):
        """Test getting an expired session (lines 467->470)."""
        manager = SessionManager()
        session_id = str(uuid.uuid4())
        session = manager.get_or_create_session(session_id)
        session = manager.sessions[session_id]

        # Mock session as expired
        with patch.object(session, "is_expired", return_value=True):
            retrieved_session = manager.get_or_create_session(session_id)

            assert retrieved_session is None
            assert session_id in manager.sessions_to_cleanup

    @pytest.mark.asyncio
    async def test_remove_session_exists(self):
        """Test removing an existing session (lines 474-479)."""
        manager = SessionManager()
        session_id = str(uuid.uuid4())
        session = manager.get_or_create_session(session_id)

        # Mock the session's clear method
        session = manager.sessions[session_id]
        with patch.object(session, "clear", new_callable=AsyncMock) as mock_clear:
            result = await manager.remove_session(session_id)

            assert result is True
            assert session_id not in manager.sessions
            mock_clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_session_nonexistent(self):
        """Test removing a non-existent session."""
        manager = SessionManager()
        result = await manager.remove_session("nonexistent-id")
        assert result is False

    def test_list_sessions_with_data(self):
        """Test listing sessions that have data (lines 483-484)."""
        manager = SessionManager()

        # Create sessions
        session1_id = str(uuid.uuid4())
        session2_id = str(uuid.uuid4())
        manager.get_or_create_session(session1_id)
        manager.get_or_create_session(session2_id)

        # Load data into one session
        session1 = manager.sessions[session1_id]
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        session1.load_data(df, "test.csv")

        # Mock has_data method
        with (
            patch.object(session1, "has_data", return_value=True),
            patch.object(manager.sessions[session2_id], "has_data", return_value=False),
        ):
            session_list = manager.list_sessions()

            assert len(session_list) == 1
            assert session_list[0].session_id == session1_id

    def test_list_sessions_cleanup_expired(self):
        """Test that list_sessions triggers cleanup of expired sessions."""
        manager = SessionManager()

        with patch.object(manager, "_cleanup_expired") as mock_cleanup:
            manager.list_sessions()
            mock_cleanup.assert_called_once()

    def test_cleanup_expired_sessions(self):
        """Test _cleanup_expired marks expired sessions for cleanup."""
        manager = SessionManager()
        session1_id = str(uuid.uuid4())
        session2_id = str(uuid.uuid4())
        manager.get_or_create_session(session1_id)
        manager.get_or_create_session(session2_id)

        # Mock one session as expired
        with (
            patch.object(manager.sessions[session1_id], "is_expired", return_value=True),
            patch.object(manager.sessions[session2_id], "is_expired", return_value=False),
            patch("databeak.core.session.logger") as mock_logger,
        ):
            manager._cleanup_expired()

            assert session1_id in manager.sessions_to_cleanup
            assert session2_id not in manager.sessions_to_cleanup
            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_marked_sessions(self):
        """Test cleanup_marked_sessions method (lines 499-501)."""
        manager = SessionManager()
        session1_id = str(uuid.uuid4())
        session2_id = str(uuid.uuid4())
        manager.get_or_create_session(session1_id)
        manager.get_or_create_session(session2_id)

        # Mark sessions for cleanup
        manager.sessions_to_cleanup.add(session1_id)
        manager.sessions_to_cleanup.add("nonexistent-id")

        with patch.object(manager, "remove_session", new_callable=AsyncMock) as mock_remove:
            await manager.cleanup_marked_sessions()

            # Should try to remove both marked sessions
            assert mock_remove.call_count == 2
            assert len(manager.sessions_to_cleanup) == 0


class TestMemoryConfiguration:
    """Test memory threshold configuration functionality."""

    def test_memory_threshold_configuration(self):
        """Test that memory threshold is configurable via settings."""
        settings = DataBeakSettings(memory_threshold_mb=4096)
        assert settings.memory_threshold_mb == 4096

    @pytest.mark.asyncio
    async def test_environment_variable_configuration(self):
        """Test that memory settings can be configured via environment variables."""
        import os

        # Set environment variables
        old_memory = os.environ.get("DATABEAK_MEMORY_THRESHOLD_MB")

        try:
            os.environ["DATABEAK_MEMORY_THRESHOLD_MB"] = "4096"

            # Create new settings instance to pick up env vars
            settings = DataBeakSettings()

            assert settings.memory_threshold_mb == 4096

        finally:
            # Clean up environment variables
            if old_memory is not None:
                os.environ["DATABEAK_MEMORY_THRESHOLD_MB"] = old_memory
            else:
                os.environ.pop("DATABEAK_MEMORY_THRESHOLD_MB", None)
