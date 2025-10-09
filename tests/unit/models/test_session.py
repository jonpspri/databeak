"""Unit tests for session.py module."""

import uuid
from datetime import UTC
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from databeak.core.session import (
    DatabeakSession,
    SessionManager,
    get_session_manager,
)
from databeak.core.settings import DatabeakSettings


class TestDatabeakSettings:
    """Tests for DatabeakSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings initialization."""
        settings = DatabeakSettings()
        assert settings.session_timeout == 3600
        assert settings.health_memory_threshold_mb == 2048
        assert settings.max_anomaly_sample_size == 10000  # Anomaly detection sample size


class TestDatabeakSession:
    """Tests for DatabeakSession class functionality."""

    def test_df_property_setter_and_getter(self) -> None:
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

    def test_df_property_deleter(self) -> None:
        """Test DataFrame property deleter (lines 109-113)."""
        session = DatabeakSession()
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        session.df = df

        # Verify data is there
        assert session.df is not None

        # Test deleter
        del session.df
        assert session.df is None

    def test_has_data_method(self) -> None:
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


class TestSessionManager:
    """Tests for SessionManager functionality."""

    def test_get_session_manager(self) -> None:
        """Test getting session manager instance."""
        manager = get_session_manager()
        assert manager is not None
        # Singleton pattern
        manager2 = get_session_manager()
        assert manager is manager2

    def test_session_manager_init(self) -> None:
        """Test SessionManager initialization."""
        manager = SessionManager(max_sessions=10, ttl_minutes=30)
        assert manager.max_sessions == 10
        assert manager.ttl_minutes == 30
        assert len(manager.sessions) == 0
        assert len(manager.sessions_to_cleanup) == 0

    def test_get_session_creates_new(self) -> None:
        """Test get_session creates new session when needed."""
        manager = SessionManager()
        session_id = str(uuid.uuid4())
        _session = manager.get_or_create_session(session_id)

        assert session_id is not None
        assert session_id in manager.sessions
        assert len(manager.sessions) == 1

    def test_get_session_max_sessions_limit(self) -> None:
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

    def test_get_session_valid(self) -> None:
        """Test getting a valid, non-expired session."""
        manager = SessionManager()
        session_id = str(uuid.uuid4())
        _session = manager.get_or_create_session(session_id)

        retrieved_session = manager.get_or_create_session(session_id)
        assert retrieved_session is not None
        assert retrieved_session.session_id == session_id

    def test_get_session_nonexistent(self) -> None:
        """Test getting a non-existent session."""
        manager = SessionManager()
        # get_session creates if not exists, so check sessions dict directly
        retrieved_session = manager.sessions.get("nonexistent-id")
        assert retrieved_session is None

    @pytest.mark.skip(reason="Session expiration logic needs clarification")
    def test_get_session_expired(self) -> None:
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
    async def test_remove_session_exists(self) -> None:
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
    async def test_remove_session_nonexistent(self) -> None:
        """Test removing a non-existent session."""
        manager = SessionManager()
        result = await manager.remove_session("nonexistent-id")
        assert result is False

    def test_list_sessions_with_data(self) -> None:
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

    def test_list_sessions_cleanup_expired(self) -> None:
        """Test that list_sessions triggers cleanup of expired sessions."""
        manager = SessionManager()

        with patch.object(manager, "_cleanup_expired") as mock_cleanup:
            manager.list_sessions()
            mock_cleanup.assert_called_once()

    def test_cleanup_expired_sessions(self) -> None:
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
    async def test_cleanup_marked_sessions(self) -> None:
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

    def test_memory_threshold_configuration(self) -> None:
        """Test that memory threshold is configurable via settings."""
        settings = DatabeakSettings(health_memory_threshold_mb=4096)
        assert settings.health_memory_threshold_mb == 4096

    @pytest.mark.asyncio
    async def test_environment_variable_configuration(self) -> None:
        """Test that memory settings can be configured via environment variables."""
        import os

        # Set environment variables
        old_memory = os.environ.get("DATABEAK_HEALTH_MEMORY_THRESHOLD_MB")

        try:
            os.environ["DATABEAK_HEALTH_MEMORY_THRESHOLD_MB"] = "4096"

            # Create new settings instance to pick up env vars
            settings = DatabeakSettings()

            assert settings.health_memory_threshold_mb == 4096

        finally:
            # Clean up environment variables
            if old_memory is not None:
                os.environ["DATABEAK_HEALTH_MEMORY_THRESHOLD_MB"] = old_memory
            else:
                os.environ.pop("DATABEAK_HEALTH_MEMORY_THRESHOLD_MB", None)
