"""Tests for history server."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp.exceptions import ToolError

from src.databeak.servers.history_server import (
    # Data models
    AutoSaveConfig,
    AutoSaveConfigResult,
    AutoSaveDisableResult,
    AutoSaveStatusResult,
    ClearHistoryResult,
    ExportHistoryResult,
    HistoryOperation,
    HistoryResult,
    HistorySummary,
    ManualSaveResult,
    RedoResult,
    RestoreResult,
    # Response models
    UndoResult,
    clear_history,
    # Auto-save operations
    configure_auto_save,
    disable_auto_save,
    export_history,
    get_auto_save_status,
    get_history,
    redo_operation,
    restore_to_operation,
    trigger_manual_save,
    # History operations
    undo_operation,
)
from tests.test_mock_context import create_mock_context


@pytest.mark.skip(
    reason="TODO: Mock/async compatibility issues - complex history functionality needs real session implementation"
)
class TestHistoryOperations:
    """Test history operation functions."""

    @pytest.mark.asyncio
    async def test_undo_operation_success(self):
        """Test successful undo operation."""
        mock_session = Mock()
        mock_session.undo = AsyncMock(
            return_value={
                "success": True,
                "message": "Operation undone",
                "operation_type": "filter",
                "previous_operation": "sort",
            }
        )
        mock_session.get_history.return_value = {
            "can_undo": True,
            "can_redo": False,
            "current_position": 2,
        }

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with patch(
            "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
        ):
            result = await undo_operation(create_mock_context())

        assert isinstance(result, UndoResult)
        assert result.success is True
        assert result.operation_undone == "filter"
        assert result.previous_operation == "sort"
        assert result.can_undo_more is True
        assert result.can_redo is False
        assert result.history_position == 2

        mock_session.undo.assert_called_once()

    @pytest.mark.skip(
        reason="TODO: get_or_create_session never returns None - need to redesign session not found behavior"
    )
    @pytest.mark.asyncio
    async def test_undo_operation_session_not_found(self):
        """Test undo operation with missing session."""
        mock_manager = Mock()
        mock_manager.get_session.return_value = None

        with (
            patch(
                "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
            ),
            pytest.raises(ToolError, match="Session 'test_session' not found"),
        ):
            await undo_operation(create_mock_context())

    @pytest.mark.asyncio
    async def test_undo_operation_failure(self):
        """Test undo operation failure."""
        mock_session = Mock()
        mock_session.undo = AsyncMock(
            return_value={
                "success": False,
                "error": "No operations to undo",
            }
        )

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with (
            patch(
                "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
            ),
            pytest.raises(ToolError, match="Undo operation failed: No operations to undo"),
        ):
            await undo_operation(create_mock_context())

    @pytest.mark.asyncio
    async def test_redo_operation_success(self):
        """Test successful redo operation."""
        mock_session = Mock()
        mock_session.redo = AsyncMock(
            return_value={
                "success": True,
                "message": "Operation redone",
                "operation_type": "transform",
                "next_operation": "validate",
            }
        )
        mock_session.get_history.return_value = {
            "can_undo": False,
            "can_redo": True,
            "current_position": 1,
        }

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with patch(
            "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
        ):
            result = await redo_operation(create_mock_context())

        assert isinstance(result, RedoResult)
        assert result.success is True
        assert result.operation_redone == "transform"
        assert result.next_operation == "validate"
        assert result.can_undo is False
        assert result.can_redo_more is True
        assert result.history_position == 1

        mock_session.redo.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_history_success(self):
        """Test successful history retrieval."""
        mock_operations = [
            {
                "operation_id": "op1",
                "operation_type": "load",
                "timestamp": "2024-01-01T12:00:00",
                "description": "Load CSV data",
                "can_undo": True,
                "can_redo": False,
            },
            {
                "operation_id": "op2",
                "operation_type": "filter",
                "timestamp": "2024-01-01T12:01:00",
                "description": "Filter rows",
                "can_undo": False,
                "can_redo": True,
            },
        ]

        mock_session = Mock()
        mock_session.get_history.return_value = {
            "success": True,
            "operations": mock_operations,
            "total_operations": 2,
            "can_undo": True,
            "can_redo": False,
            "current_position": 1,
            "history_enabled": True,
        }

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with patch(
            "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
        ):
            result = await get_history(create_mock_context(), limit=10)

        assert isinstance(result, HistoryResult)
        assert result.success is True
        assert len(result.operations) == 2
        assert result.operations[0].operation_id == "op1"
        assert result.operations[0].operation_type == "load"
        assert result.operations[1].operation_id == "op2"
        assert result.operations[1].operation_type == "filter"
        assert result.summary.total_operations == 2
        assert result.summary.can_undo is True
        assert result.summary.can_redo is False
        assert result.summary.current_position == 1
        assert result.summary.history_enabled is True
        assert result.total_found == 2
        assert result.limit_applied == 10

        mock_session.get_history.assert_called_once_with(10)

    @pytest.mark.asyncio
    async def test_restore_to_operation_success(self):
        """Test successful restore to operation."""
        mock_session = Mock()
        mock_session.restore_to_operation = AsyncMock(
            return_value={
                "success": True,
                "operations_undone": 3,
                "operations_redone": 1,
                "final_position": 5,
            }
        )

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with patch(
            "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
        ):
            result = await restore_to_operation(create_mock_context(), "target_op")

        assert isinstance(result, RestoreResult)
        assert result.success is True
        assert result.restored_to_operation == "target_op"
        assert result.operations_undone == 3
        assert result.operations_redone == 1
        assert result.final_position == 5

        mock_session.restore_to_operation.assert_called_once_with("target_op")

    @pytest.mark.asyncio
    async def test_clear_history_success(self):
        """Test successful history clearing."""
        mock_history_manager = Mock()

        mock_session = Mock()
        mock_session.history_manager = mock_history_manager
        mock_session.get_history.return_value = {"total_operations": 5}

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with patch(
            "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
        ):
            result = await clear_history(create_mock_context())

        assert isinstance(result, ClearHistoryResult)
        assert result.success is True
        assert result.operations_cleared == 5
        assert result.history_was_enabled is True

        mock_history_manager.clear_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_history_no_history_manager(self):
        """Test clear history with no history manager."""
        mock_session = Mock()
        mock_session.history_manager = None

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with (
            patch(
                "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
            ),
            pytest.raises(ToolError, match="History is not enabled for this session"),
        ):
            await clear_history(create_mock_context())

    @pytest.mark.asyncio
    async def test_export_history_success(self):
        """Test successful history export."""
        mock_history_manager = Mock()
        mock_history_manager.export_history.return_value = True

        mock_session = Mock()
        mock_session.history_manager = mock_history_manager
        mock_session.get_history.return_value = {"total_operations": 10}

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with (
            patch(
                "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
            ),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 1024
            result = await export_history(create_mock_context(), "/tmp/history.json", format="json")

        assert isinstance(result, ExportHistoryResult)
        assert result.success is True
        assert result.file_path == "/tmp/history.json"
        assert result.format == "json"
        assert result.operations_exported == 10
        assert result.file_size_bytes == 1024

        mock_history_manager.export_history.assert_called_once_with("/tmp/history.json", "json")

    @pytest.mark.asyncio
    async def test_export_history_failure(self):
        """Test history export failure."""
        mock_history_manager = Mock()
        mock_history_manager.export_history.return_value = False

        mock_session = Mock()
        mock_session.history_manager = mock_history_manager

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with (
            patch(
                "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
            ),
            pytest.raises(ToolError, match="History export operation failed"),
        ):
            await export_history(create_mock_context(), "/tmp/history.json")


@pytest.mark.skip(
    reason="TODO: Mock/async compatibility issues - complex auto-save functionality needs real session implementation"
)
class TestAutoSaveOperations:
    """Test auto-save operation functions."""

    @pytest.mark.asyncio
    async def test_configure_auto_save_success(self):
        """Test successful auto-save configuration."""
        mock_session = Mock()
        mock_session.get_auto_save_status.return_value = {"enabled": False}
        mock_session.enable_auto_save = AsyncMock(return_value={"success": True})

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with patch(
            "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
        ):
            result = await configure_auto_save(
                create_mock_context(),
                enabled=True,
                mode="after_operation",
                strategy="backup",
                max_backups=5,
                format="csv",
            )

        assert isinstance(result, AutoSaveConfigResult)
        assert result.success is True
        assert result.config.enabled is True
        assert result.config.mode == "after_operation"
        assert result.config.strategy == "backup"
        assert result.config.max_backups == 5
        assert result.config.format == "csv"
        assert result.config_changed is True

        # Verify the session method was called with correct config
        call_args = mock_session.enable_auto_save.call_args[0][0]
        assert call_args["enabled"] is True
        assert call_args["mode"] == "after_operation"
        assert call_args["strategy"] == "backup"
        assert call_args["max_backups"] == 5

    @pytest.mark.asyncio
    async def test_configure_auto_save_with_previous_config(self):
        """Test auto-save configuration with previous config."""
        previous_config_dict = {
            "enabled": True,
            "mode": "periodic",
            "strategy": "overwrite",
            "format": "csv",
            "encoding": "utf-8",
        }

        mock_session = Mock()
        mock_session.get_auto_save_status.return_value = {
            "enabled": True,
            "config": previous_config_dict,
        }
        mock_session.enable_auto_save = AsyncMock(return_value={"success": True})

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with patch(
            "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
        ):
            result = await configure_auto_save(
                create_mock_context(),
                enabled=True,
                mode="after_operation",  # Different from previous
                strategy="backup",  # Different from previous
            )

        assert isinstance(result, AutoSaveConfigResult)
        assert result.previous_config is not None
        assert result.previous_config.mode == "periodic"
        assert result.previous_config.strategy == "overwrite"
        assert result.config_changed is True

    @pytest.mark.asyncio
    async def test_configure_auto_save_validation_error(self):
        """Test auto-save configuration with validation error."""
        mock_manager = Mock()
        mock_manager.get_session.return_value = Mock()

        with (
            patch(
                "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
            ),
            pytest.raises(ToolError),
        ):
            await configure_auto_save(
                create_mock_context(),
                interval_seconds=10,  # Too low, should trigger validation error
            )

    @pytest.mark.asyncio
    async def test_disable_auto_save_success(self):
        """Test successful auto-save disable."""
        mock_session = Mock()
        mock_session.get_auto_save_status.return_value = {"enabled": True}
        mock_session.disable_auto_save = AsyncMock(
            return_value={
                "success": True,
                "final_save_performed": True,
                "final_save_path": "/tmp/final_save.csv",
            }
        )

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with patch(
            "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
        ):
            result = await disable_auto_save(create_mock_context())

        assert isinstance(result, AutoSaveDisableResult)
        assert result.success is True
        assert result.was_enabled is True
        assert result.final_save_performed is True
        assert result.final_save_path == "/tmp/final_save.csv"

        mock_session.disable_auto_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_disable_auto_save_already_disabled(self):
        """Test disable auto-save when already disabled."""
        mock_session = Mock()
        mock_session.get_auto_save_status.return_value = {"enabled": False}
        mock_session.disable_auto_save = AsyncMock(
            return_value={
                "success": True,
                "final_save_performed": False,
            }
        )

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with patch(
            "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
        ):
            result = await disable_auto_save(create_mock_context())

        assert result.was_enabled is False
        assert result.final_save_performed is False

    @pytest.mark.asyncio
    async def test_get_auto_save_status_enabled(self):
        """Test get auto-save status when enabled."""
        status_dict = {
            "enabled": True,
            "config": {
                "enabled": True,
                "mode": "after_operation",
                "strategy": "backup",
                "format": "csv",
                "encoding": "utf-8",
                "max_backups": 10,
            },
            "last_save_time": "2024-01-01T12:00:00",
            "save_count": 5,
            "last_save_path": "/tmp/backup.csv",
            "next_scheduled_save": "2024-01-01T12:30:00",
        }

        mock_session = Mock()
        mock_session.get_auto_save_status.return_value = status_dict

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with patch(
            "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
        ):
            result = await get_auto_save_status(create_mock_context())

        assert isinstance(result, AutoSaveStatusResult)
        assert result.success is True
        assert result.status.enabled is True
        assert result.status.config is not None
        assert result.status.config.mode == "after_operation"
        assert result.status.config.strategy == "backup"
        assert result.status.config.max_backups == 10
        assert result.status.last_save_time == "2024-01-01T12:00:00"
        assert result.status.save_count == 5
        assert result.status.last_save_path == "/tmp/backup.csv"
        assert result.status.next_scheduled_save == "2024-01-01T12:30:00"

    @pytest.mark.asyncio
    async def test_get_auto_save_status_disabled(self):
        """Test get auto-save status when disabled."""
        status_dict = {"enabled": False}

        mock_session = Mock()
        mock_session.get_auto_save_status.return_value = status_dict

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with patch(
            "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
        ):
            result = await get_auto_save_status(create_mock_context())

        assert result.status.enabled is False
        assert result.status.config is None

    @pytest.mark.asyncio
    async def test_trigger_manual_save_success(self):
        """Test successful manual save."""
        # Mock DataFrame for session
        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=100)  # 100 rows
        mock_df.columns = ["col1", "col2", "col3"]  # 3 columns

        mock_session = Mock()
        mock_session.has_data.return_value = True
        mock_session.df = mock_df
        mock_session.manual_save = AsyncMock(
            return_value={
                "success": True,
                "save_path": "/tmp/manual_save.csv",
                "format": "csv",
                "file_size_bytes": 2048,
                "save_time": "2024-01-01T12:00:00",
            }
        )

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with patch(
            "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
        ):
            result = await trigger_manual_save(create_mock_context())

        assert isinstance(result, ManualSaveResult)
        assert result.success is True
        assert result.save_path == "/tmp/manual_save.csv"
        assert result.format == "csv"
        assert result.rows_saved == 100
        assert result.columns_saved == 3
        assert result.file_size_bytes == 2048
        assert result.save_time == "2024-01-01T12:00:00"

        mock_session.manual_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_manual_save_no_data(self):
        """Test manual save with no data loaded."""
        mock_session = Mock()
        mock_session.has_data.return_value = False
        mock_session.df = None
        mock_session.manual_save = AsyncMock(
            return_value={
                "success": True,
                "save_path": "/tmp/empty.csv",
                "format": "csv",
            }
        )

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with patch(
            "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
        ):
            result = await trigger_manual_save(create_mock_context())

        assert result.rows_saved == 0
        assert result.columns_saved == 0

    @pytest.mark.asyncio
    async def test_trigger_manual_save_failure(self):
        """Test manual save failure."""
        mock_session = Mock()
        mock_session.manual_save = AsyncMock(
            return_value={
                "success": False,
                "error": "Disk full",
            }
        )

        mock_manager = Mock()
        mock_manager.get_session.return_value = mock_session

        with (
            patch(
                "src.databeak.servers.history_server.get_session_manager", return_value=mock_manager
            ),
            pytest.raises(ToolError, match="Manual save failed: Disk full"),
        ):
            await trigger_manual_save(create_mock_context())


class TestAutoSaveConfigValidation:
    """Test AutoSaveConfig Pydantic model validation."""

    def test_valid_config(self):
        """Test valid auto-save configuration."""
        config = AutoSaveConfig(
            enabled=True,
            mode="hybrid",
            strategy="versioned",
            interval_seconds=300,
            max_backups=20,
            format="json",
        )

        assert config.enabled is True
        assert config.mode == "hybrid"
        assert config.strategy == "versioned"
        assert config.interval_seconds == 300
        assert config.max_backups == 20
        assert config.format == "json"

    def test_interval_validation(self):
        """Test interval validation."""
        # Valid interval
        config = AutoSaveConfig(
            enabled=True, mode="periodic", strategy="backup", interval_seconds=60
        )
        assert config.interval_seconds == 60

        # Invalid interval - too low
        with pytest.raises(ValueError, match="Interval must be at least 30 seconds"):
            AutoSaveConfig(enabled=True, mode="periodic", strategy="backup", interval_seconds=10)

    def test_max_backups_validation(self):
        """Test max_backups validation."""
        # Valid max_backups
        config = AutoSaveConfig(
            enabled=True, mode="after_operation", strategy="backup", max_backups=5
        )
        assert config.max_backups == 5

        # Invalid max_backups
        with pytest.raises(ValueError, match="Maximum backups must be at least 1"):
            AutoSaveConfig(enabled=True, mode="after_operation", strategy="backup", max_backups=0)

    def test_defaults(self):
        """Test default values."""
        config = AutoSaveConfig(enabled=True, mode="after_operation", strategy="backup")

        assert config.format == "csv"
        assert config.encoding == "utf-8"
        assert config.interval_seconds is None
        assert config.max_backups is None
        assert config.backup_dir is None
        assert config.custom_path is None


class TestHistoryDataModels:
    """Test history-related data models."""

    def test_history_operation_model(self):
        """Test HistoryOperation model."""
        op = HistoryOperation(
            operation_id="op123",
            operation_type="transform",
            timestamp="2024-01-01T12:00:00",
            description="Applied data transformation",
            can_undo=True,
            can_redo=False,
        )

        assert op.operation_id == "op123"
        assert op.operation_type == "transform"
        assert op.can_undo is True
        assert op.can_redo is False

    def test_history_summary_model(self):
        """Test HistorySummary model."""
        summary = HistorySummary(
            total_operations=5,
            can_undo=True,
            can_redo=False,
            current_position=3,
            history_enabled=True,
        )

        assert summary.total_operations == 5
        assert summary.can_undo is True
        assert summary.can_redo is False
        assert summary.current_position == 3
        assert summary.history_enabled is True
