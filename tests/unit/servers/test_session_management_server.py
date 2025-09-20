"""Tests for the unified session management server."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastmcp.exceptions import ToolError

from src.databeak.servers.session_management_server import (
    AutoSaveConfig,
    clear_session_history,
    configure_auto_save,
    disable_auto_save,
    get_auto_save_status,
    get_session_history,
    redo_operation,
    session_management_server,
    trigger_manual_save,
    undo_operation,
)
from tests.test_mock_context import create_mock_context


class TestSessionManagementServer:
    """Test unified session management server setup."""

    def test_server_exists(self):
        """Test that session management server is properly configured."""
        assert session_management_server is not None
        assert session_management_server.name == "DataBeak-SessionManagement"
        instructions = session_management_server.instructions or ""
        assert "session management" in instructions.lower()

    def test_server_has_unified_functionality(self):
        """Test that server consolidates both history and auto-save operations."""
        # Verify the server is properly configured
        assert hasattr(session_management_server, "name")
        assert hasattr(session_management_server, "instructions")

        # Should mention both history and auto-save in instructions
        instructions = (session_management_server.instructions or "").lower()
        assert "history" in instructions
        assert "auto-save" in instructions


class TestAutoSaveConfig:
    """Test the unified AutoSaveConfig model."""

    def test_valid_config_creation(self):
        """Test creating valid AutoSaveConfig."""
        config = AutoSaveConfig(
            enabled=True,
            mode="after_operation",
            strategy="versioned",
            file_path="/tmp/test.csv",
            format="csv",
            interval_seconds=300,
            encoding="utf-8",
            backup_count=5,
        )

        assert config.enabled is True
        assert config.mode == "after_operation"
        assert config.strategy == "versioned"
        assert config.interval_seconds == 300

    def test_interval_validation(self):
        """Test interval validation in AutoSaveConfig."""
        with pytest.raises(ValueError, match="Auto-save interval must be at least 10 seconds"):
            AutoSaveConfig(
                enabled=True,
                mode="periodic",
                strategy="overwrite",
                interval_seconds=5,  # Too low
            )

    def test_backup_count_validation(self):
        """Test backup count validation in AutoSaveConfig."""
        with pytest.raises(ValueError, match="Backup count must be at least 1"):
            AutoSaveConfig(
                enabled=True,
                mode="after_operation",
                strategy="backup",
                backup_count=0,  # Too low
            )


class TestHistoryOperations:
    """Test history management operations."""

    @pytest.mark.asyncio
    async def test_undo_operation_success(self):
        """Test successful undo operation."""
        with patch(
            "src.databeak.servers.session_management_server.get_session_manager"
        ) as mock_manager:
            mock_session = Mock()
            mock_session.undo = AsyncMock(
                return_value={
                    "success": True,
                    "message": "Undid filter operation",
                    "operation": {"type": "filter", "details": {}},
                    "can_undo": False,
                    "can_redo": True,
                }
            )

            mock_session_manager = Mock()
            mock_session_manager.get_session.return_value = mock_session
            mock_manager.return_value = mock_session_manager

            result = await undo_operation(create_mock_context(), "test-session")

            assert result.success is True
            assert result.message == "Undid filter operation"
            assert result.can_redo is True
            mock_session.undo.assert_called_once()

    @pytest.mark.asyncio
    async def test_undo_operation_failure(self):
        """Test undo operation failure."""
        with patch(
            "src.databeak.servers.session_management_server.get_session_manager"
        ) as mock_manager:
            mock_session = Mock()
            mock_session.undo = AsyncMock(
                return_value={"success": False, "error": "No operations to undo"}
            )

            mock_session_manager = Mock()
            mock_session_manager.get_session.return_value = mock_session
            mock_manager.return_value = mock_session_manager

            result = await undo_operation(create_mock_context(), "test-session")

            assert result.success is False
            assert "No operations to undo" in result.message

    @pytest.mark.asyncio
    async def test_redo_operation_success(self):
        """Test successful redo operation."""
        with patch(
            "src.databeak.servers.session_management_server.get_session_manager"
        ) as mock_manager:
            mock_session = Mock()
            mock_session.redo = AsyncMock(
                return_value={
                    "success": True,
                    "message": "Redid sort operation",
                    "operation": {"type": "sort", "details": {}},
                    "can_undo": True,
                    "can_redo": False,
                }
            )

            mock_session_manager = Mock()
            mock_session_manager.get_session.return_value = mock_session
            mock_manager.return_value = mock_session_manager

            result = await redo_operation(create_mock_context(), "test-session")

            assert result.success is True
            assert result.message == "Redid sort operation"
            assert result.can_undo is True
            mock_session.redo.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_history_success(self):
        """Test successful history retrieval."""
        with patch(
            "src.databeak.servers.session_management_server.get_session_manager"
        ) as mock_manager:
            mock_session = Mock()
            mock_session.get_history = Mock(
                return_value={
                    "success": True,
                    "history": [
                        {"type": "load", "timestamp": "2023-01-01T00:00:00"},
                        {"type": "filter", "timestamp": "2023-01-01T00:01:00"},
                    ],
                    "total": 2,
                }
            )

            mock_session_manager = Mock()
            mock_session_manager.get_session.return_value = mock_session
            mock_manager.return_value = mock_session_manager

            result = await get_session_history(create_mock_context(), "test-session", limit=10)

            assert result.success is True
            assert len(result.operations) == 2
            assert result.total_operations == 2
            mock_session.get_history.assert_called_once_with(10)

    @pytest.mark.skip(reason="TODO: Fix mock setup for operations_history list behavior")
    @pytest.mark.asyncio
    async def test_clear_session_history_success(self):
        """Test successful history clearing."""
        with patch(
            "src.databeak.servers.session_management_server.get_session_manager"
        ) as mock_manager:
            # Create a mock session with proper list for operations_history
            mock_session = Mock()
            mock_session.history_manager = Mock()
            mock_session.history_manager.clear_history = Mock()

            # Mock operations_history to behave like a list
            mock_operations = MagicMock()
            mock_operations.__len__.return_value = 3
            mock_operations.clear = Mock()
            mock_session.operations_history = mock_operations

            mock_session_manager = Mock()
            mock_session_manager.get_session.return_value = mock_session
            mock_manager.return_value = mock_session_manager

            result = await clear_session_history(create_mock_context(), "test-session")

            assert result.success is True
            assert "3 operations" in result.message
            assert result.session_id == "test-session"
            assert result.operation_type == "clear_history"
            mock_session.history_manager.clear_history.assert_called_once()


class TestAutoSaveOperations:
    """Test auto-save management operations."""

    @pytest.mark.skip(reason="TODO: Fix AsyncMock setup for enable_auto_save method")
    @pytest.mark.asyncio
    async def test_configure_auto_save_success(self):
        """Test successful auto-save configuration."""
        with patch(
            "src.databeak.servers.session_management_server.get_session_manager"
        ) as mock_manager:
            mock_session = Mock()
            mock_session.enable_auto_save = AsyncMock(
                return_value={"success": True, "message": "Auto-save enabled"}
            )

            mock_session_manager = Mock()
            mock_session_manager.get_session.return_value = mock_session
            mock_manager.return_value = mock_session_manager

            config = AutoSaveConfig(
                enabled=True,
                mode="after_operation",
                strategy="versioned",
                interval_seconds=300,
            )

            result = await configure_auto_save(create_mock_context(), "test-session", config)

            assert result.success is True
            assert result.config == config
            mock_session.enable_auto_save.assert_called_once()

    @pytest.mark.skip(reason="TODO: Fix AsyncMock setup for auto-save disable operations")
    @pytest.mark.asyncio
    async def test_disable_auto_save_with_final_save(self):
        """Test disabling auto-save with final save."""
        with patch(
            "src.databeak.servers.session_management_server.get_session_manager"
        ) as mock_manager:
            mock_session = Mock()
            mock_session.get_auto_save_status = Mock(return_value={"enabled": True})
            mock_session.manual_save = AsyncMock(return_value={"success": True})
            mock_session.disable_auto_save = AsyncMock(return_value={"success": True})

            mock_session_manager = Mock()
            mock_session_manager.get_session.return_value = mock_session
            mock_manager.return_value = mock_session_manager

            result = await disable_auto_save(
                create_mock_context(), "test-session", perform_final_save=True
            )

            assert result.success is True
            assert result.was_enabled is True
            assert result.final_save_performed is True
            mock_session.manual_save.assert_called_once()
            mock_session.disable_auto_save.assert_called_once()

    @pytest.mark.skip(reason="TODO: Fix AutoSaveStatus validation for mock data")
    @pytest.mark.asyncio
    async def test_get_auto_save_status_success(self):
        """Test successful auto-save status retrieval."""
        with patch(
            "src.databeak.servers.session_management_server.get_session_manager"
        ) as mock_manager:
            mock_session = Mock()
            mock_session.get_auto_save_status = Mock(
                return_value={
                    "enabled": True,
                    "last_save_time": "2023-01-01T00:00:00",
                    "last_save_success": True,
                }
            )

            mock_session_manager = Mock()
            mock_session_manager.get_session.return_value = mock_session
            mock_manager.return_value = mock_session_manager

            result = await get_auto_save_status(create_mock_context(), "test-session")

            assert result.success is True
            assert result.status.enabled is True
            assert result.status.last_save_time == "2023-01-01T00:00:00"

    @pytest.mark.skip(reason="TODO: Fix AsyncMock setup for manual_save method")
    @pytest.mark.asyncio
    async def test_trigger_manual_save_success(self):
        """Test successful manual save trigger."""
        with patch(
            "src.databeak.servers.session_management_server.get_session_manager"
        ) as mock_manager:
            mock_session = Mock()
            mock_session.manual_save = AsyncMock(
                return_value={
                    "success": True,
                    "file_path": "/tmp/test.csv",
                    "rows": 100,
                    "columns": 5,
                    "timestamp": "2023-01-01T00:00:00",
                }
            )

            mock_session_manager = Mock()
            mock_session_manager.get_session.return_value = mock_session
            mock_manager.return_value = mock_session_manager

            result = await trigger_manual_save(create_mock_context(), "test-session")

            assert result.success is True
            assert result.file_path == "/tmp/test.csv"
            assert result.rows_saved == 100
            assert result.columns_saved == 5


class TestErrorHandling:
    """Test error handling in session management operations."""

    @pytest.mark.asyncio
    async def test_session_not_found_error(self):
        """Test handling of session not found errors."""
        with patch(
            "src.databeak.servers.session_management_server.get_session_manager"
        ) as mock_manager:
            mock_session_manager = Mock()
            mock_session_manager.get_session.return_value = None
            mock_manager.return_value = mock_session_manager

            with pytest.raises(ToolError, match="Session .* not found"):
                await undo_operation(create_mock_context(), "nonexistent-session")

    @pytest.mark.asyncio
    async def test_unexpected_error_handling(self):
        """Test handling of unexpected errors."""
        with patch(
            "src.databeak.servers.session_management_server.get_session_manager"
        ) as mock_manager:
            mock_manager.side_effect = Exception("Unexpected error")

            with pytest.raises(ToolError, match="Undo operation failed"):
                await undo_operation(create_mock_context(), "test-session")

    @pytest.mark.asyncio
    async def test_context_logging(self):
        """Test that operations use FastMCP context for logging."""
        mock_ctx = AsyncMock()

        with patch(
            "src.databeak.servers.session_management_server.get_session_manager"
        ) as mock_manager:
            mock_session = Mock()
            mock_session.get_history = Mock(
                return_value={"success": True, "history": [], "total": 0}
            )

            mock_session_manager = Mock()
            mock_session_manager.get_session.return_value = mock_session
            mock_manager.return_value = mock_session_manager

            await get_session_history(mock_ctx, "test-session")

            # Verify context was used for logging
            mock_ctx.info.assert_called()
