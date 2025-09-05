"""Tests for uncovered session management functionality."""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from src.databeak.models.auto_save import AutoSaveConfig, AutoSaveStrategy
from src.databeak.models.csv_session import CSVSession, get_csv_settings
from src.databeak.models.data_models import OperationType
from src.databeak.models.history_manager import HistoryStorage


class TestCSVSessionAutoSave:
    """Test auto-save functionality in CSV sessions."""

    def test_load_data_sets_original_file_path(self):
        """Test that loading data with file_path sets auto_save_manager.original_file_path."""
        session = CSVSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        file_path = "/test/data.csv"

        session.load_data(df, file_path)

        # This should hit line 106 - setting original_file_path
        assert session.auto_save_manager.original_file_path == file_path

    def test_load_data_without_file_path(self):
        """Test that loading data without file_path doesn't set original_file_path."""
        session = CSVSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

        session.load_data(df, None)

        # Should not set original_file_path when file_path is None
        assert session.auto_save_manager.original_file_path is None

    @pytest.mark.asyncio
    async def test_trigger_auto_save_when_needed(self):
        """Test auto-save triggering when configured and needed."""
        auto_save_config = AutoSaveConfig(enabled=True, strategy=AutoSaveStrategy.OVERWRITE)
        session = CSVSession(auto_save_config=auto_save_config)
        df = pd.DataFrame({"name": ["Alice"], "age": [25]})

        # Mock the auto_save_manager methods
        session.auto_save_manager.should_save_after_operation = Mock(return_value=True)
        session.auto_save_manager.trigger_save = AsyncMock(return_value={"success": True})

        # Load data and mark as needing save
        session.load_data(df, "test.csv")
        session.data_session.metadata["needs_autosave"] = True

        # Test auto-save trigger
        result = await session.trigger_auto_save_if_needed()

        # Should call trigger_save and return result
        assert result is not None
        assert result["success"] is True
        session.auto_save_manager.trigger_save.assert_called_once()
        # Should clear the autosave flag
        assert session.data_session.metadata["needs_autosave"] is False

    @pytest.mark.asyncio
    async def test_trigger_auto_save_not_needed(self):
        """Test auto-save when not configured or not needed."""
        session = CSVSession()
        df = pd.DataFrame({"name": ["Alice"], "age": [25]})
        session.load_data(df, "test.csv")

        # Mock should_save_after_operation to return False
        session.auto_save_manager.should_save_after_operation = Mock(return_value=False)

        result = await session.trigger_auto_save_if_needed()

        # Should return None when auto-save not needed
        assert result is None


class TestCSVSessionHistoryIntegration:
    """Test history manager integration in CSV sessions."""

    def test_record_operation_with_history_enabled(self):
        """Test recording operations when history is enabled."""
        session = CSVSession(enable_history=True)
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.load_data(df, "test.csv")

        # Mock the history manager
        session.history_manager.add_operation = Mock()

        # Record an operation
        operation_details = {"test": "data"}
        session.record_operation(OperationType.FILTER, operation_details)

        # Verify history manager was called (lines 145-156)
        session.history_manager.add_operation.assert_called_once()
        call_args = session.history_manager.add_operation.call_args
        assert call_args[1]["operation_type"] == "filter"
        assert call_args[1]["details"] == operation_details
        assert call_args[1]["current_data"] is session.data_session.df

    def test_record_operation_with_history_disabled(self):
        """Test recording operations when history is disabled."""
        session = CSVSession(enable_history=False)
        df = pd.DataFrame({"name": ["Alice"], "age": [25]})
        session.load_data(df, "test.csv")

        # Record an operation
        operation_details = {"test": "data"}
        initial_history_length = len(session.operations_history)
        session.record_operation(OperationType.FILTER, operation_details)

        # Verify legacy history was updated but no history manager called
        assert len(session.operations_history) == initial_history_length + 1  # just filter
        assert session.history_manager is None

    def test_record_operation_with_string_operation_type(self):
        """Test recording operations with string operation type."""
        session = CSVSession(enable_history=True)
        df = pd.DataFrame({"name": ["Alice"], "age": [25]})
        session.load_data(df, "test.csv")

        # Mock history manager
        session.history_manager.add_operation = Mock()

        # Use string instead of OperationType enum
        session.record_operation("custom_operation", {"test": "data"})

        # Should handle string operation types (lines 130-132)
        session.history_manager.add_operation.assert_called()
        call_args = session.history_manager.add_operation.call_args
        assert call_args[1]["operation_type"] == "custom_operation"

    def test_record_operation_marks_autosave_needed(self):
        """Test that recording operations marks auto-save as needed."""
        session = CSVSession()
        df = pd.DataFrame({"name": ["Alice"], "age": [25]})
        session.load_data(df, "test.csv")

        # Clear the autosave flag
        session.data_session.metadata["needs_autosave"] = False

        session.record_operation(OperationType.FILTER, {"test": "data"})

        # Should mark auto-save as needed (line 159)
        assert session.data_session.metadata["needs_autosave"] is True


class TestCSVSessionLifecycle:
    """Test session lifecycle management."""

    def test_session_expiration_check(self):
        """Test session expiration checking."""
        # Create session with very short TTL
        session = CSVSession(ttl_minutes=0.01)  # 0.6 seconds

        # Should not be expired immediately
        assert not session.is_expired()

        # Mock the lifecycle to be expired
        session.lifecycle.is_expired = Mock(return_value=True)
        assert session.is_expired()

    def test_update_access_time_updates_both_components(self):
        """Test that update_access_time updates both lifecycle and data_session."""
        session = CSVSession()

        # Mock both update methods
        session.lifecycle.update_access_time = Mock()
        session.data_session.update_access_time = Mock()

        session.update_access_time()

        # Both should be called
        session.lifecycle.update_access_time.assert_called_once()
        session.data_session.update_access_time.assert_called_once()


class TestCSVSessionInitialization:
    """Test session initialization with various configurations."""

    def test_session_with_custom_session_id(self):
        """Test session creation with custom session ID."""
        custom_id = "test-session-123"
        session = CSVSession(session_id=custom_id)

        assert session.session_id == custom_id

    def test_session_with_auto_generated_id(self):
        """Test session creation with auto-generated ID."""
        session = CSVSession()

        assert session.session_id is not None
        assert len(session.session_id) > 0
        assert isinstance(session.session_id, str)

    def test_session_with_custom_auto_save_config(self, tmp_path):
        """Test session creation with custom auto-save configuration."""
        backup_dir = str(tmp_path / "backups")
        custom_config = AutoSaveConfig(
            enabled=True, strategy=AutoSaveStrategy.BACKUP, backup_dir=backup_dir
        )
        session = CSVSession(auto_save_config=custom_config)

        assert session.auto_save_config == custom_config
        assert session.auto_save_manager.config == custom_config

    def test_session_with_history_disabled(self):
        """Test session creation with history disabled."""
        session = CSVSession(enable_history=False)

        assert session.enable_history is False
        assert session.history_manager is None

    def test_session_with_different_history_storage(self):
        """Test session creation with different history storage types."""
        session = CSVSession(enable_history=True, history_storage=HistoryStorage.PICKLE)

        assert session.enable_history is True
        assert session.history_manager is not None
        assert session.history_manager.storage_type == HistoryStorage.PICKLE


class TestCSVSessionSettings:
    """Test CSV session integration with settings."""

    def test_session_uses_global_settings(self):
        """Test that session uses global settings for history directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create custom settings and store original value
            custom_settings = get_csv_settings()
            original_history_dir = custom_settings.csv_history_dir

            try:
                custom_settings.csv_history_dir = temp_dir

                session = CSVSession(enable_history=True)

                # Session should use the global settings
                assert session.history_manager is not None
                # The history manager should be configured with the settings directory
                assert session.history_manager.history_dir == temp_dir
            finally:
                # Restore original settings to avoid affecting other tests
                custom_settings.csv_history_dir = original_history_dir

    def test_session_info_structure(self):
        """Test that get_info returns properly structured SessionInfo."""
        session = CSVSession()
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "city": ["NYC", "LA", "Chicago"],
            }
        )
        session.load_data(df, "test_data.csv")

        info = session.get_info()

        # Verify all required fields
        assert info.session_id == session.session_id
        assert info.row_count == 3
        assert info.column_count == 3
        assert info.columns == ["name", "age", "city"]
        assert info.file_path == "test_data.csv"
        assert info.operations_count >= 1  # At least the load operation
        assert info.created_at is not None
        assert info.last_accessed is not None
        assert info.memory_usage_mb >= 0
        # Memory usage should be calculated (could be 0.0 for small DataFrames due to rounding)
        assert isinstance(info.memory_usage_mb, float)


class TestSessionManagerIntegration:
    """Test SessionManager functionality."""

    def test_session_manager_creation_and_retrieval(self):
        """Test creating and retrieving sessions through the manager."""
        from src.databeak.models.csv_session import get_session_manager

        manager = get_session_manager()

        # Create session
        new_session_id = manager.create_session()
        assert new_session_id is not None

        # Retrieve session
        retrieved = manager.get_session(new_session_id)
        assert retrieved is not None
        assert retrieved.session_id == new_session_id

    def test_session_manager_cleanup_expired_sessions(self):
        """Test that session manager can clean up expired sessions."""
        from src.databeak.models.csv_session import get_session_manager

        manager = get_session_manager()

        # Create session (manager controls TTL)
        session_id = manager.create_session()
        session = manager.get_session(session_id)

        # Mock expiration
        session.lifecycle.is_expired = Mock(return_value=True)

        # Use the correct cleanup method
        manager._cleanup_expired()

        # Session should be removed
        assert manager.get_session(session_id) is None
