"""Unit tests for auto-save operations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.databeak.tools.auto_save_operations import (
    configure_auto_save,
    disable_auto_save,
    get_auto_save_status,
    trigger_manual_save,
)


class TestConfigureAutoSave:
    """Tests for configure_auto_save function."""

    @pytest.mark.asyncio
    async def test_configure_auto_save_success(self):
        """Test successful auto-save configuration."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            # Mock session manager and session
            mock_manager = MagicMock()
            mock_session = MagicMock()
            mock_session.enable_auto_save = AsyncMock(
                return_value={
                    "success": True,
                    "config": {"enabled": True, "mode": "after_operation", "strategy": "backup"},
                }
            )
            mock_manager.get_session.return_value = mock_session
            mock_get_manager.return_value = mock_manager

            result = await configure_auto_save("test-session")

            assert result["success"] is True
            assert result["session_id"] == "test-session"
            assert "data" in result
            mock_session.enable_auto_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_configure_auto_save_session_not_found(self):
        """Test auto-save configuration when session not found."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_session.return_value = None
            mock_get_manager.return_value = mock_manager

            result = await configure_auto_save("invalid-session")

            assert result["success"] is False
            assert "Session not found" in result["message"]
            assert "invalid-session" in result["error"]

    @pytest.mark.asyncio
    async def test_configure_auto_save_with_all_parameters(self):
        """Test auto-save configuration with all parameters."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_session = MagicMock()
            mock_session.enable_auto_save = AsyncMock(return_value={"success": True, "config": {}})
            mock_manager.get_session.return_value = mock_session
            mock_get_manager.return_value = mock_manager

            result = await configure_auto_save(
                "test-session",
                enabled=False,
                mode="periodic",
                strategy="versioned",
                interval_seconds=600,
                max_backups=5,
                backup_dir="/tmp/backups",
                custom_path="/custom/path.csv",
                format="json",
                encoding="utf-16",
            )

            assert result["success"] is True
            # Check that enable_auto_save was called with correct config
            call_args = mock_session.enable_auto_save.call_args[0][0]
            assert call_args["enabled"] is False
            assert call_args["mode"] == "periodic"
            assert call_args["strategy"] == "versioned"
            assert call_args["interval_seconds"] == 600
            assert call_args["max_backups"] == 5
            assert call_args["backup_dir"] == "/tmp/backups"
            assert call_args["custom_path"] == "/custom/path.csv"
            assert call_args["format"] == "json"
            assert call_args["encoding"] == "utf-16"

    @pytest.mark.asyncio
    async def test_configure_auto_save_enable_fails(self):
        """Test auto-save configuration when enable_auto_save fails."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_session = MagicMock()
            mock_session.enable_auto_save = AsyncMock(
                return_value={"success": False, "error": "Configuration invalid"}
            )
            mock_manager.get_session.return_value = mock_session
            mock_get_manager.return_value = mock_manager

            result = await configure_auto_save("test-session")

            assert result["success"] is False
            assert "Failed to configure auto-save" in result["message"]
            assert result["error"] == "Configuration invalid"

    @pytest.mark.asyncio
    async def test_configure_auto_save_with_context(self):
        """Test auto-save configuration with FastMCP context."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_session = MagicMock()
            mock_session.enable_auto_save = AsyncMock(return_value={"success": True, "config": {}})
            mock_manager.get_session.return_value = mock_session
            mock_get_manager.return_value = mock_manager

            mock_ctx = AsyncMock()

            result = await configure_auto_save("test-session", ctx=mock_ctx)

            assert result["success"] is True
            mock_ctx.info.assert_called()

    @pytest.mark.asyncio
    async def test_configure_auto_save_exception(self):
        """Test auto-save configuration with exception."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_session.side_effect = Exception("Database error")
            mock_get_manager.return_value = mock_manager

            result = await configure_auto_save("test-session")

            assert result["success"] is False
            assert "Failed to configure auto-save" in result["message"]
            assert "Database error" in result["error"]


class TestDisableAutoSave:
    """Tests for disable_auto_save function."""

    @pytest.mark.asyncio
    async def test_disable_auto_save_success(self):
        """Test successful auto-save disabling."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_session = MagicMock()
            mock_session.disable_auto_save = AsyncMock(return_value={"success": True})
            mock_manager.get_session.return_value = mock_session
            mock_get_manager.return_value = mock_manager

            result = await disable_auto_save("test-session")

            assert result["success"] is True
            assert result["session_id"] == "test-session"
            assert "Auto-save disabled" in result["message"]
            mock_session.disable_auto_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_disable_auto_save_session_not_found(self):
        """Test auto-save disabling when session not found."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_session.return_value = None
            mock_get_manager.return_value = mock_manager

            result = await disable_auto_save("invalid-session")

            assert result["success"] is False
            assert "Session not found" in result["message"]

    @pytest.mark.asyncio
    async def test_disable_auto_save_fails(self):
        """Test auto-save disabling when disable fails."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_session = MagicMock()
            mock_session.disable_auto_save = AsyncMock(
                return_value={"success": False, "error": "Already disabled"}
            )
            mock_manager.get_session.return_value = mock_session
            mock_get_manager.return_value = mock_manager

            result = await disable_auto_save("test-session")

            assert result["success"] is False
            assert "Failed to disable auto-save" in result["message"]
            assert result["error"] == "Already disabled"

    @pytest.mark.asyncio
    async def test_disable_auto_save_with_context(self):
        """Test auto-save disabling with FastMCP context."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_session = MagicMock()
            mock_session.disable_auto_save = AsyncMock(return_value={"success": True})
            mock_manager.get_session.return_value = mock_session
            mock_get_manager.return_value = mock_manager

            mock_ctx = AsyncMock()

            result = await disable_auto_save("test-session", ctx=mock_ctx)

            assert result["success"] is True
            mock_ctx.info.assert_called()

    @pytest.mark.asyncio
    async def test_disable_auto_save_exception(self):
        """Test auto-save disabling with exception."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_session.side_effect = Exception("Network error")
            mock_get_manager.return_value = mock_manager

            mock_ctx = AsyncMock()

            result = await disable_auto_save("test-session", ctx=mock_ctx)

            assert result["success"] is False
            assert "Failed to disable auto-save" in result["message"]
            mock_ctx.error.assert_called_once()


class TestGetAutoSaveStatus:
    """Tests for get_auto_save_status function."""

    @pytest.mark.asyncio
    async def test_get_auto_save_status_success(self):
        """Test successful auto-save status retrieval."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_session = MagicMock()
            status_data = {
                "enabled": True,
                "mode": "after_operation",
                "strategy": "backup",
                "last_save": "2023-01-01T10:00:00Z",
            }
            mock_session.get_auto_save_status.return_value = status_data
            mock_manager.get_session.return_value = mock_session
            mock_get_manager.return_value = mock_manager

            result = await get_auto_save_status("test-session")

        assert result["success"] is True
        assert result["session_id"] == "test-session"
        assert result["data"] == status_data
        assert "Auto-save status retrieved" in result["message"]

    @pytest.mark.asyncio
    async def test_get_auto_save_status_session_not_found(self):
        """Test auto-save status retrieval when session not found."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_session.return_value = None
            mock_get_manager.return_value = mock_manager

            result = await get_auto_save_status("invalid-session")

        assert result["success"] is False
        assert "Session not found" in result["message"]

    @pytest.mark.asyncio
    async def test_get_auto_save_status_with_context(self):
        """Test auto-save status retrieval with FastMCP context."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_session = MagicMock()
            mock_session.get_auto_save_status.return_value = {"enabled": False}
            mock_manager.get_session.return_value = mock_session
            mock_get_manager.return_value = mock_manager

            mock_ctx = AsyncMock()

            result = await get_auto_save_status("test-session", ctx=mock_ctx)

        assert result["success"] is True
        mock_ctx.info.assert_called()

    @pytest.mark.asyncio
    async def test_get_auto_save_status_exception(self):
        """Test auto-save status retrieval with exception."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_session.side_effect = Exception("Status error")
            mock_get_manager.return_value = mock_manager

            mock_ctx = AsyncMock()

            result = await get_auto_save_status("test-session", ctx=mock_ctx)

        assert result["success"] is False
        assert "Failed to get auto-save status" in result["message"]
        mock_ctx.error.assert_called_once()


class TestTriggerManualSave:
    """Tests for trigger_manual_save function."""

    @pytest.mark.asyncio
    async def test_trigger_manual_save_success(self):
        """Test successful manual save trigger."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_session = MagicMock()
            save_result = {
                "success": True,
                "save_path": "/path/to/saved/file.csv",
                "timestamp": "2023-01-01T10:00:00Z",
            }
            mock_session.manual_save = AsyncMock(return_value=save_result)
            mock_manager.get_session.return_value = mock_session
            mock_get_manager.return_value = mock_manager

            result = await trigger_manual_save("test-session")

        assert result["success"] is True
        assert result["session_id"] == "test-session"
        assert result["data"] == save_result
        assert "Manual save completed" in result["message"]
        mock_session.manual_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_manual_save_session_not_found(self):
        """Test manual save trigger when session not found."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_session.return_value = None
            mock_get_manager.return_value = mock_manager

            result = await trigger_manual_save("invalid-session")

        assert result["success"] is False
        assert "Session not found" in result["message"]

    @pytest.mark.asyncio
    async def test_trigger_manual_save_fails(self):
        """Test manual save trigger when save fails."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_session = MagicMock()
            mock_session.manual_save = AsyncMock(
                return_value={"success": False, "error": "Disk full"}
            )
            mock_manager.get_session.return_value = mock_session
            mock_get_manager.return_value = mock_manager

            result = await trigger_manual_save("test-session")

        assert result["success"] is False
        assert "Manual save failed" in result["message"]
        assert result["error"] == "Disk full"

    @pytest.mark.asyncio
    async def test_trigger_manual_save_with_context(self):
        """Test manual save trigger with FastMCP context."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_session = MagicMock()
            mock_session.manual_save = AsyncMock(
                return_value={"success": True, "save_path": "/test/path.csv"}
            )
            mock_manager.get_session.return_value = mock_session
            mock_get_manager.return_value = mock_manager

            mock_ctx = AsyncMock()

            result = await trigger_manual_save("test-session", ctx=mock_ctx)

        assert result["success"] is True
        assert mock_ctx.info.call_count == 2  # Called twice: trigger + completion

    @pytest.mark.asyncio
    async def test_trigger_manual_save_exception(self):
        """Test manual save trigger with exception."""
        with patch(
            "src.databeak.tools.auto_save_operations.get_session_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_session.side_effect = Exception("Save error")
            mock_get_manager.return_value = mock_manager

            mock_ctx = AsyncMock()

            result = await trigger_manual_save("test-session", ctx=mock_ctx)

        assert result["success"] is False
        assert "Failed to trigger manual save" in result["message"]
        mock_ctx.error.assert_called_once()
