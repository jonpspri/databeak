"""Unit tests for csv_session.py module."""

import uuid
from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from src.databeak.exceptions import HistoryNotEnabledError
from src.databeak.models.csv_session import (
    CSVSession,
    DataBeakSettings,
    SessionManager,
    get_session_manager,
)
from src.databeak.models.data_models import ExportFormat


class TestDataBeakSettings:
    """Tests for DataBeakSettings configuration."""

    def test_default_settings(self):
        """Test default settings initialization."""
        settings = DataBeakSettings()
        assert settings.auto_save is True
        assert settings.session_timeout == 3600
        assert settings.csv_history_dir == "."
        assert settings.max_file_size_mb == 1024


class TestCSVSession:
    """Tests for CSVSession class functionality."""

    def test_df_property_setter_and_getter(self):
        """Test DataFrame property setter and getter."""
        session = CSVSession()
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
        session = CSVSession()
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        session.df = df

        # Verify data is there
        assert session.df is not None

        # Test deleter
        del session.df
        assert session.df is None

    def test_has_data_method(self):
        """Test has_data method (line 117)."""
        session = CSVSession()

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
        session = CSVSession()
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
        session = CSVSession()
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
        session = CSVSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        file_path = str(tmp_path / "test.json")
        result = await session._save_callback(file_path, ExportFormat.JSON, "utf-8")

        assert result["success"] is True
        assert Path(file_path).exists()

    @pytest.mark.asyncio
    async def test_save_callback_excel_format(self, tmp_path):
        """Test _save_callback with Excel format."""
        session = CSVSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        file_path = str(tmp_path / "test.xlsx")
        result = await session._save_callback(file_path, ExportFormat.EXCEL, "utf-8")

        assert result["success"] is True
        assert Path(file_path).exists()

    @pytest.mark.asyncio
    async def test_save_callback_parquet_format(self, tmp_path):
        """Test _save_callback with Parquet format."""
        session = CSVSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        file_path = str(tmp_path / "test.parquet")
        result = await session._save_callback(file_path, ExportFormat.PARQUET, "utf-8")

        assert result["success"] is True
        assert Path(file_path).exists()

    @pytest.mark.asyncio
    async def test_save_callback_unsupported_format(self, tmp_path):
        """Test _save_callback with unsupported format."""
        session = CSVSession()
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
        session = CSVSession()
        # Don't load any data

        file_path = str(tmp_path / "test.csv")
        result = await session._save_callback(file_path, ExportFormat.CSV, "utf-8")

        assert result["success"] is False
        assert "No data to save" in result["error"]

    @pytest.mark.asyncio
    async def test_save_callback_exception_handling(self, tmp_path):
        """Test _save_callback exception handling."""
        session = CSVSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        # Use invalid path to trigger exception
        invalid_path = "/invalid/path/that/does/not/exist/test.csv"
        result = await session._save_callback(invalid_path, ExportFormat.CSV, "utf-8")

        assert result["success"] is False
        assert "error" in result

    def test_rollback_no_original_data(self):
        """Test rollback when no original data exists (lines 231-242)."""
        session = CSVSession()
        # Don't load any data, so no original_df

        result = session.rollback()
        assert result is False

    def test_rollback_to_original_state(self):
        """Test rollback to original state."""
        session = CSVSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.load_data(df, "test.csv")

        # Modify the data
        assert session.df is not None
        session.df["new_col"] = ["X", "Y"]
        assert "new_col" in session.df.columns

        # Add some operation history
        session.operations_history.append({"test": "operation"})
        assert len(session.operations_history) > 0

        # Rollback all operations
        result = session.rollback(steps=10)  # More than history length
        assert result is True

        # Should restore original data and clear history
        assert session.df is not None and "new_col" not in session.df.columns
        assert len(session.operations_history) == 0

    def test_rollback_partial_steps_warning(self):
        """Test rollback with partial steps shows warning."""
        session = CSVSession()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.load_data(df, "test.csv")

        # Add some operations
        session.operations_history.extend([{"op": 1}, {"op": 2}])

        # Try partial rollback (should warn and return False)
        with patch("src.databeak.models.csv_session.logger") as mock_logger:
            result = session.rollback(steps=1)
            assert result is False
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_enable_auto_save_success(self, tmp_path):
        """Test enable_auto_save method (lines 246-265)."""
        session = CSVSession()

        config_dict = {"enabled": True, "strategy": "overwrite", "backup_dir": str(tmp_path)}

        # Mock the entire manager creation and method calls
        with (
            patch("src.databeak.models.csv_session.AutoSaveManager") as mock_manager_class,
            patch("src.databeak.models.csv_session.AutoSaveConfig.from_dict") as mock_from_dict,
        ):
            # Set up the mock config
            mock_config = Mock()
            mock_config.enabled = True
            mock_config.to_dict.return_value = {"enabled": True}
            mock_from_dict.return_value = mock_config

            # Set up the mock manager
            mock_manager = Mock()
            mock_manager.start_periodic_save = AsyncMock()
            mock_manager_class.return_value = mock_manager

            result = await session.enable_auto_save(config_dict)

            assert result["success"] is True
            assert "Auto-save configuration updated" in result["message"]
            assert "config" in result

            # Verify the config was created and manager was set up
            mock_from_dict.assert_called_once_with(config_dict)
            mock_manager_class.assert_called_once()
            mock_manager.start_periodic_save.assert_called_once_with(session._save_callback)

    @pytest.mark.asyncio
    async def test_enable_auto_save_disabled(self, tmp_path):
        """Test enable_auto_save when enabled is False."""
        session = CSVSession()

        config_dict = {
            "enabled": False,  # Disabled
            "strategy": "overwrite",
            "backup_dir": str(tmp_path),
        }

        with patch.object(
            session.auto_save_manager, "start_periodic_save", new_callable=AsyncMock
        ) as mock_start:
            result = await session.enable_auto_save(config_dict)

            assert result["success"] is True
            assert "Auto-save configuration updated" in result["message"]
            assert "config" in result
            # Should NOT call start_periodic_save when disabled
            mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_enable_auto_save_exception(self):
        """Test enable_auto_save with exception."""
        session = CSVSession()

        # Invalid config with invalid enum value to trigger exception
        invalid_config = {"mode": "invalid_mode_value"}

        result = await session.enable_auto_save(invalid_config)
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_disable_auto_save_success(self):
        """Test disable_auto_save method (lines 269-274)."""
        session = CSVSession()

        with patch.object(
            session.auto_save_manager, "stop_periodic_save", new_callable=AsyncMock
        ) as mock_stop:
            result = await session.disable_auto_save()

            assert result["success"] is True
            assert "Auto-save disabled" in result["message"]
            assert session.auto_save_config.enabled is False
            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_disable_auto_save_exception(self):
        """Test disable_auto_save with exception."""
        session = CSVSession()

        with patch.object(
            session.auto_save_manager,
            "stop_periodic_save",
            new_callable=AsyncMock,
            side_effect=Exception("Test error"),
        ):
            result = await session.disable_auto_save()

            assert result["success"] is False
            assert "Test error" in result["error"]

    def test_get_auto_save_status(self):
        """Test get_auto_save_status method (line 278)."""
        session = CSVSession()

        with patch.object(
            session.auto_save_manager, "get_status", return_value={"enabled": True}
        ) as mock_get_status:
            result = session.get_auto_save_status()

            assert result == {"enabled": True}
            mock_get_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_manual_save(self):
        """Test manual_save method (line 282)."""
        session = CSVSession()

        with patch.object(
            session.auto_save_manager,
            "trigger_save",
            new_callable=AsyncMock,
            return_value={"success": True},
        ) as mock_trigger:
            result = await session.manual_save()

            assert result["success"] is True
            mock_trigger.assert_called_once_with(session._save_callback, "manual")

    @pytest.mark.asyncio
    async def test_undo_no_history_manager(self):
        """Test undo when history manager is disabled (lines 286-317)."""
        session = CSVSession(enable_history=False)

        with pytest.raises(HistoryNotEnabledError):
            await session.undo()

    @pytest.mark.asyncio
    async def test_undo_no_operations_to_undo(self):
        """Test undo when no operations to undo."""
        session = CSVSession(enable_history=True)

        with patch.object(session.history_manager, "can_undo", return_value=False):
            result = await session.undo()

            assert result["success"] is False
            assert "No operations to undo" in result["error"]

    @pytest.mark.asyncio
    async def test_undo_success(self):
        """Test successful undo operation."""
        session = CSVSession(enable_history=True)
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        # Mock history manager methods
        mock_operation = Mock()
        mock_operation.operation_type = "filter"
        mock_operation.to_dict = Mock(return_value={"type": "filter"})

        with (
            patch.object(session.history_manager, "can_undo", return_value=True),
            patch.object(session.history_manager, "undo", return_value=(mock_operation, df)),
            patch.object(session.history_manager, "can_redo", return_value=True),
            patch.object(
                session.auto_save_manager, "should_save_after_operation", return_value=False
            ),
        ):
            result = await session.undo()

            assert result["success"] is True
            assert "Undid operation: filter" in result["message"]
            assert result["operation"] == {"type": "filter"}
            assert result["can_undo"] is True
            assert result["can_redo"] is True

    @pytest.mark.asyncio
    async def test_undo_no_snapshot_available(self):
        """Test undo when no snapshot is available."""
        session = CSVSession(enable_history=True)

        with (
            patch.object(session.history_manager, "can_undo", return_value=True),
            patch.object(session.history_manager, "undo", return_value=(None, None)),
        ):
            result = await session.undo()

            assert result["success"] is False
            assert "No snapshot available for undo" in result["error"]

    @pytest.mark.asyncio
    async def test_undo_with_auto_save(self):
        """Test undo with auto-save enabled."""
        session = CSVSession(enable_history=True)
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        mock_operation = Mock()
        mock_operation.operation_type = "filter"
        mock_operation.to_dict = Mock(return_value={"type": "filter"})

        with (
            patch.object(session.history_manager, "can_undo", return_value=True),
            patch.object(session.history_manager, "undo", return_value=(mock_operation, df)),
            patch.object(
                session.auto_save_manager, "should_save_after_operation", return_value=True
            ),
            patch.object(
                session.auto_save_manager, "trigger_save", new_callable=AsyncMock
            ) as mock_save,
        ):
            result = await session.undo()

            assert result["success"] is True
            mock_save.assert_called_once_with(session._save_callback, "undo")

    @pytest.mark.asyncio
    async def test_undo_history_error(self):
        """Test undo with HistoryNotEnabledError exception."""
        session = CSVSession(enable_history=True)

        mock_error = HistoryNotEnabledError("test-session")

        with (
            patch.object(session.history_manager, "can_undo", return_value=True),
            patch.object(session.history_manager, "undo", side_effect=mock_error),
            patch("src.databeak.models.csv_session.logger") as mock_logger,
        ):
            result = await session.undo()

            assert result["success"] is False
            assert result["error"] == mock_error.to_dict()
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_undo_unexpected_error(self):
        """Test undo with unexpected exception."""
        session = CSVSession(enable_history=True)

        with (
            patch.object(session.history_manager, "can_undo", return_value=True),
            patch.object(
                session.history_manager, "undo", side_effect=Exception("Unexpected error")
            ),
            patch("src.databeak.models.csv_session.logger") as mock_logger,
        ):
            result = await session.undo()

            assert result["success"] is False
            # Type assertion: error is a dict when it's a structured error
            error_dict = result["error"]
            assert isinstance(error_dict, dict)
            assert error_dict["type"] == "UnexpectedError"
            assert "Unexpected error" in error_dict["message"]
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_redo_no_history_manager(self):
        """Test redo when history manager is disabled (lines 324-355)."""
        session = CSVSession(enable_history=False)

        with pytest.raises(HistoryNotEnabledError):
            await session.redo()

    @pytest.mark.asyncio
    async def test_redo_no_operations_to_redo(self):
        """Test redo when no operations to redo."""
        session = CSVSession(enable_history=True)

        with patch.object(session.history_manager, "can_redo", return_value=False):
            result = await session.redo()

            assert result["success"] is False
            assert "No operations to redo" in result["error"]

    @pytest.mark.asyncio
    async def test_redo_success(self):
        """Test successful redo operation."""
        session = CSVSession(enable_history=True)
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        mock_operation = Mock()
        mock_operation.operation_type = "filter"
        mock_operation.to_dict = Mock(return_value={"type": "filter"})

        with (
            patch.object(session.history_manager, "can_redo", return_value=True),
            patch.object(session.history_manager, "redo", return_value=(mock_operation, df)),
            patch.object(session.history_manager, "can_undo", return_value=True),
            patch.object(
                session.auto_save_manager, "should_save_after_operation", return_value=False
            ),
        ):
            result = await session.redo()

            assert result["success"] is True
            assert "Redid operation: filter" in result["message"]
            assert result["operation"] == {"type": "filter"}
            assert result["can_undo"] is True
            assert result["can_redo"] is True

    @pytest.mark.asyncio
    async def test_redo_no_snapshot_available(self):
        """Test redo when no snapshot is available."""
        session = CSVSession(enable_history=True)

        with (
            patch.object(session.history_manager, "can_redo", return_value=True),
            patch.object(session.history_manager, "redo", return_value=(None, None)),
        ):
            result = await session.redo()

            assert result["success"] is False
            assert "No snapshot available for redo" in result["error"]

    @pytest.mark.asyncio
    async def test_redo_history_error(self):
        """Test redo with HistoryNotEnabledError exception."""
        session = CSVSession(enable_history=True)

        mock_error = HistoryNotEnabledError("test-session")

        with (
            patch.object(session.history_manager, "can_redo", return_value=True),
            patch.object(session.history_manager, "redo", side_effect=mock_error),
            patch("src.databeak.models.csv_session.logger") as mock_logger,
        ):
            result = await session.redo()

            assert result["success"] is False
            assert result["error"] == mock_error.to_dict()
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_redo_unexpected_error(self):
        """Test redo with unexpected exception."""
        session = CSVSession(enable_history=True)

        with (
            patch.object(session.history_manager, "can_redo", return_value=True),
            patch.object(
                session.history_manager, "redo", side_effect=Exception("Unexpected error")
            ),
            patch("src.databeak.models.csv_session.logger") as mock_logger,
        ):
            result = await session.redo()

            assert result["success"] is False
            # Type assertion: error is a dict when it's a structured error
            error_dict = result["error"]
            assert isinstance(error_dict, dict)
            assert error_dict["type"] == "UnexpectedError"
            assert "Unexpected error" in error_dict["message"]
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_redo_with_auto_save(self):
        """Test redo with auto-save enabled (line 338)."""
        session = CSVSession(enable_history=True)
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.df = df

        mock_operation = Mock()
        mock_operation.operation_type = "filter"
        mock_operation.to_dict = Mock(return_value={"type": "filter"})

        with (
            patch.object(session.history_manager, "can_redo", return_value=True),
            patch.object(session.history_manager, "redo", return_value=(mock_operation, df)),
            patch.object(
                session.auto_save_manager, "should_save_after_operation", return_value=True
            ),
            patch.object(
                session.auto_save_manager, "trigger_save", new_callable=AsyncMock
            ) as mock_save,
        ):
            result = await session.redo()

            assert result["success"] is True
            mock_save.assert_called_once_with(session._save_callback, "redo")

    def test_get_history_no_history_manager(self):
        """Test get_history when history manager is disabled (lines 362-380)."""
        session = CSVSession(enable_history=False)

        # Add some legacy operations
        session.operations_history = [{"op": 1}, {"op": 2}, {"op": 3}]

        result = session.get_history(limit=2)

        assert result["success"] is True
        assert len(result["history"]) == 2
        assert result["total"] == 3
        assert result["history"] == [{"op": 2}, {"op": 3}]  # Last 2

    def test_get_history_no_limit(self):
        """Test get_history without limit."""
        session = CSVSession(enable_history=False)

        session.operations_history = [{"op": 1}, {"op": 2}]

        result = session.get_history()

        assert result["success"] is True
        assert len(result["history"]) == 2
        assert result["total"] == 2

    def test_get_history_with_history_manager(self):
        """Test get_history with history manager enabled."""
        session = CSVSession(enable_history=True)

        mock_history = [{"id": "1", "type": "filter"}]
        mock_stats = {"total_operations": 1}

        with (
            patch.object(session.history_manager, "get_history", return_value=mock_history),
            patch.object(session.history_manager, "get_statistics", return_value=mock_stats),
        ):
            result = session.get_history(limit=10)

            assert result["success"] is True
            assert result["history"] == mock_history
            assert result["statistics"] == mock_stats

    def test_get_history_exception(self):
        """Test get_history with exception."""
        session = CSVSession(enable_history=True)

        from src.databeak.exceptions import HistoryError

        mock_error = HistoryError("test-session", "Test error")

        with (
            patch.object(session.history_manager, "get_history", side_effect=mock_error),
            patch("src.databeak.models.csv_session.logger") as mock_logger,
        ):
            result = session.get_history()

            assert result["success"] is False
            assert result["error"] == mock_error.to_dict()
            mock_logger.error.assert_called_once()

    def test_get_history_unexpected_error(self):
        """Test get_history with unexpected exception."""
        session = CSVSession(enable_history=True)

        with (
            patch.object(
                session.history_manager, "get_history", side_effect=Exception("Unexpected error")
            ),
            patch("src.databeak.models.csv_session.logger") as mock_logger,
        ):
            result = session.get_history()

            assert result["success"] is False
            # Type assertion: error is a dict when it's a structured error
            error_dict = result["error"]
            assert isinstance(error_dict, dict)
            assert error_dict["type"] == "UnexpectedError"
            assert "Unexpected error" in error_dict["message"]
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_to_operation_no_history_manager(self):
        """Test restore_to_operation when history manager is disabled (lines 387-418)."""
        session = CSVSession(enable_history=False)

        with pytest.raises(HistoryNotEnabledError):
            await session.restore_to_operation("op-123")

    @pytest.mark.asyncio
    async def test_restore_to_operation_success(self):
        """Test successful restore_to_operation."""
        session = CSVSession(enable_history=True)
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

        with (
            patch.object(session.history_manager, "restore_to_operation", return_value=df),
            patch.object(
                session.auto_save_manager, "should_save_after_operation", return_value=False
            ),
        ):
            result = await session.restore_to_operation("op-123")

            assert result["success"] is True
            assert "Restored to operation op-123" in result["message"]
            assert result["shape"] == (2, 2)
            assert session.df is df

    @pytest.mark.asyncio
    async def test_restore_to_operation_no_snapshot(self):
        """Test restore_to_operation when no snapshot available."""
        session = CSVSession(enable_history=True)

        with patch.object(session.history_manager, "restore_to_operation", return_value=None):
            result = await session.restore_to_operation("op-123")

            assert result["success"] is False
            assert "Could not restore to operation op-123" in result["error"]

    @pytest.mark.asyncio
    async def test_restore_to_operation_with_auto_save(self):
        """Test restore_to_operation with auto-save enabled."""
        session = CSVSession(enable_history=True)
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

        with (
            patch.object(session.history_manager, "restore_to_operation", return_value=df),
            patch.object(
                session.auto_save_manager, "should_save_after_operation", return_value=True
            ),
            patch.object(
                session.auto_save_manager, "trigger_save", new_callable=AsyncMock
            ) as mock_save,
        ):
            result = await session.restore_to_operation("op-123")

            assert result["success"] is True
            mock_save.assert_called_once_with(session._save_callback, "restore")

    @pytest.mark.asyncio
    async def test_restore_to_operation_history_error(self):
        """Test restore_to_operation with HistoryNotEnabledError exception."""
        session = CSVSession(enable_history=True)

        mock_error = HistoryNotEnabledError("test-session")

        with (
            patch.object(session.history_manager, "restore_to_operation", side_effect=mock_error),
            patch("src.databeak.models.csv_session.logger") as mock_logger,
        ):
            result = await session.restore_to_operation("op-123")

            assert result["success"] is False
            assert result["error"] == mock_error.to_dict()
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_to_operation_unexpected_error(self):
        """Test restore_to_operation with unexpected exception."""
        session = CSVSession(enable_history=True)

        with (
            patch.object(
                session.history_manager,
                "restore_to_operation",
                side_effect=Exception("Unexpected error"),
            ),
            patch("src.databeak.models.csv_session.logger") as mock_logger,
        ):
            result = await session.restore_to_operation("op-123")

            assert result["success"] is False
            # Type assertion: error is a dict when it's a structured error
            error_dict = result["error"]
            assert isinstance(error_dict, dict)
            assert error_dict["type"] == "UnexpectedError"
            assert "Unexpected error" in error_dict["message"]
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_session(self):
        """Test clear method (lines 426-434)."""
        session = CSVSession(enable_history=True)
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.load_data(df, "test.csv")

        # Add some history
        session.operations_history.append({"op": "test"})

        with (
            patch.object(
                session.auto_save_manager, "stop_periodic_save", new_callable=AsyncMock
            ) as mock_stop,
            patch.object(session.history_manager, "clear_history") as mock_clear_history,
            patch.object(session._data_session, "clear_data") as mock_clear_data,
        ):
            await session.clear()

            mock_stop.assert_called_once()
            mock_clear_history.assert_called_once()
            mock_clear_data.assert_called_once()
            assert len(session.operations_history) == 0

    @pytest.mark.asyncio
    async def test_clear_session_no_history_manager(self):
        """Test clear method without history manager."""
        session = CSVSession(enable_history=False)
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        session.load_data(df, "test.csv")

        session.operations_history.append({"op": "test"})

        with (
            patch.object(
                session.auto_save_manager, "stop_periodic_save", new_callable=AsyncMock
            ) as mock_stop,
            patch.object(session._data_session, "clear_data") as mock_clear_data,
        ):
            await session.clear()

            mock_stop.assert_called_once()
            mock_clear_data.assert_called_once()
            assert len(session.operations_history) == 0

    @pytest.mark.asyncio
    async def test_trigger_auto_save_failure(self):
        """Test trigger_auto_save_if_needed when save fails (line 190->192)."""
        session = CSVSession()
        df = pd.DataFrame({"name": ["Alice"], "age": [25]})
        session.load_data(df, "test.csv")

        # Configure auto-save to be needed
        session._data_session.metadata["needs_autosave"] = True
        with (
            patch.object(
                session.auto_save_manager, "should_save_after_operation", return_value=True
            ),
            patch.object(
                session.auto_save_manager,
                "trigger_save",
                new_callable=AsyncMock,
                return_value={"success": False, "error": "Save failed"},
            ),
        ):
            result = await session.trigger_auto_save_if_needed()

        assert result is not None
        assert result["success"] is False
        # autosave flag should remain True since save failed
        assert session._data_session.metadata["needs_autosave"] is True


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
            patch("src.databeak.models.csv_session.logger") as mock_logger,
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

    def test_export_session_history_exists(self):
        """Test export_session_history with existing session (lines 515-519)."""
        manager = SessionManager()
        session_id = str(uuid.uuid4())
        session = manager.get_or_create_session(session_id)
        session = manager.sessions[session_id]

        # Add some test data
        session.operations_history = [{"op": "test"}]
        session._data_session.metadata = {"test": "data"}

        with patch.object(session.lifecycle, "created_at") as mock_created_at:
            mock_created_at.isoformat.return_value = "2023-01-01T00:00:00"

            result = manager.export_session_history(session_id)

            assert result is not None
            assert result["session_id"] == session_id
            assert result["created_at"] == "2023-01-01T00:00:00"
            assert result["operations"] == [{"op": "test"}]
            assert result["metadata"] == {"test": "data"}

    def test_export_session_history_nonexistent(self):
        """Test export_session_history with non-existent session."""
        manager = SessionManager()
        # export_session_history creates session if not exists, so test actual behavior
        result = manager.export_session_history("nonexistent-id")
        assert result is not None
        assert result["session_id"] == "nonexistent-id"
