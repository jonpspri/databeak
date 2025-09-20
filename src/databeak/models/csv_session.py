"""Session management with auto-save and history features."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import Field
from pydantic_settings import BaseSettings

from ..exceptions import HistoryError, HistoryNotEnabledError
from .auto_save import AutoSaveConfig, AutoSaveManager
from .data_models import ExportFormat, OperationType, SessionInfo
from .data_session import DataSession
from .history_manager import HistoryManager, HistoryStorage
from .session_lifecycle import SessionLifecycle
from .typed_dicts import (
    AutoSaveOperationResult,
    HistoryResult,
    SessionHistoryExport,
    UndoRedoOperationResult,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class DataBeakSettings(BaseSettings):
    """Configuration settings for session management."""

    csv_history_dir: str = Field(
        default=".",
        description="Directory for storing session history files",
    )
    max_file_size_mb: int = Field(default=1024, description="Maximum file size limit in megabytes")
    session_timeout: int = Field(default=3600, description="Session timeout in seconds")
    chunk_size: int = Field(
        default=10000,
        description="Default chunk size for processing large datasets",
    )
    auto_save: bool = Field(default=True, description="Enable auto-save functionality by default")
    memory_threshold_mb: int = Field(
        default=2048, description="Memory usage threshold in MB for health monitoring"
    )
    max_history_operations: int = Field(
        default=1000, description="Maximum operations to keep in session history before cleanup"
    )
    memory_warning_threshold: float = Field(
        default=0.75, description="Memory usage ratio that triggers warning status (0.0-1.0)"
    )
    memory_critical_threshold: float = Field(
        default=0.90, description="Memory usage ratio that triggers critical status (0.0-1.0)"
    )
    session_capacity_warning_threshold: float = Field(
        default=0.90, description="Session capacity ratio that triggers warning (0.0-1.0)"
    )

    model_config = {"env_prefix": "DATABEAK_", "case_sensitive": False}


# Global settings instance
_settings: DataBeakSettings | None = None


# Implementation: Singleton pattern for global settings with environment variable support
def get_csv_settings() -> DataBeakSettings:
    """Get global DataBeak settings instance."""
    global _settings
    if _settings is None:
        _settings = DataBeakSettings()
    return _settings


class CSVSession:
    """CSV editing session with auto-save and history management."""

    # Implementation: Session initialization with TTL, auto-save, and persistent history components
    def __init__(
        self,
        session_id: str | None = None,
        ttl_minutes: int = 60,
        auto_save_config: AutoSaveConfig | None = None,
        *,
        enable_history: bool = True,
        history_storage: HistoryStorage = HistoryStorage.JSON,
    ):
        """Initialize CSV session with components."""
        self.session_id = session_id or str(uuid4())
        self.operations_history: list[dict[str, Any]] = []  # Keep for backward compatibility

        # Core components
        self._data_session = DataSession(self.session_id)
        self.lifecycle = SessionLifecycle(self.session_id, ttl_minutes)

        # Auto-save configuration
        self.auto_save_config = auto_save_config or AutoSaveConfig()
        self.auto_save_manager = AutoSaveManager(self.session_id, self.auto_save_config)

        # History management
        self.enable_history = enable_history
        settings = get_csv_settings()
        self.history_manager = (
            HistoryManager(
                session_id=self.session_id,
                storage_type=(history_storage if enable_history else HistoryStorage.MEMORY),
                history_dir=settings.csv_history_dir,
                max_history=settings.max_history_operations,
                enable_snapshots=True,
                snapshot_interval=5,  # Take snapshot every 5 operations
            )
            if enable_history
            else None
        )

    # Delegate to lifecycle manager
    def update_access_time(self) -> None:
        """Update the last accessed time."""
        self.lifecycle.update_access_time()
        self._data_session.update_access_time()

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return self.lifecycle.is_expired()

    @property
    def df(self) -> pd.DataFrame | None:
        """Get or set the DataFrame."""
        return self._data_session.df

    @df.setter
    def df(self, new_df: pd.DataFrame | None) -> None:
        """Set the DataFrame."""
        self._data_session.df = new_df
        self.update_access_time()

    @df.deleter
    def df(self) -> None:
        """Clear the DataFrame."""
        self._data_session.df = None
        self.update_access_time()

    def has_data(self) -> bool:
        """Check if data is loaded."""
        return self._data_session.has_data()

    @property
    def metadata(self) -> dict[str, Any]:
        """Get session metadata."""
        return self._data_session.metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        """Set session metadata."""
        self._data_session.metadata = value

    def load_data(self, df: pd.DataFrame, file_path: str | None = None) -> None:
        """Load data into the session."""
        self._data_session.load_data(df, file_path)
        self.update_access_time()
        self.record_operation(OperationType.LOAD, {"file_path": file_path, "shape": df.shape})

        # Update auto-save manager with original file path
        if file_path:
            self.auto_save_manager.original_file_path = file_path

    def get_info(self) -> SessionInfo:
        """Get session information."""
        data_info = self._data_session.get_data_info()
        lifecycle_info = self.lifecycle.get_lifecycle_info()

        return SessionInfo(
            session_id=self.session_id,
            created_at=lifecycle_info["created_at"],
            last_accessed=lifecycle_info["last_accessed"],
            row_count=data_info["shape"][0],
            column_count=data_info["shape"][1],
            columns=data_info["columns"],
            memory_usage_mb=data_info["memory_usage_mb"],
            operations_count=len(self.operations_history),
            file_path=data_info["file_path"],
        )

    def record_operation(
        self,
        operation_type: str | OperationType,
        details: dict[str, Any],
    ) -> None:
        """Record an operation in history."""
        # Handle both string and OperationType inputs
        operation_value = (
            operation_type.value if hasattr(operation_type, "value") else operation_type
        )

        # Legacy history (backward compatibility)
        self.operations_history.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": operation_value,
                "details": details,
            },
        )

        # Apply history limits to prevent unbounded growth
        settings = get_csv_settings()
        if len(self.operations_history) > settings.max_history_operations:
            # Keep only the most recent operations
            excess = len(self.operations_history) - settings.max_history_operations
            self.operations_history = self.operations_history[excess:]
            logger.info(
                "Trimmed %d old operations from session %s history (limit: %d)",
                excess,
                self.session_id,
                settings.max_history_operations,
            )

        self.update_access_time()

        # New persistent history
        if self.history_manager and self._data_session.df is not None:
            self.history_manager.add_operation(
                operation_type=operation_value,
                details=details,
                current_data=self._data_session.df,
                metadata={
                    "file_path": self._data_session.file_path,
                    "shape": (
                        self._data_session.df.shape if self._data_session.df is not None else (0, 0)
                    ),
                },
            )

        # Mark that auto-save is needed
        self.metadata["needs_autosave"] = True

    async def trigger_auto_save_if_needed(self) -> AutoSaveOperationResult | None:
        """Trigger auto-save after operation if configured."""
        if self.auto_save_manager.should_save_after_operation() and self.metadata.get(
            "needs_autosave",
        ):
            result = await self.auto_save_manager.trigger_save(
                self._save_callback,
                "after_operation",
            )
            if result.get("success"):
                self.metadata["needs_autosave"] = False
            return result
        return None

    async def _save_callback(
        self,
        file_path: str,
        export_format: ExportFormat,
        encoding: str,
    ) -> dict[str, Any]:
        """Callback for auto-save operations."""
        try:
            if self._data_session.df is None:
                return {"success": False, "error": "No data to save"}

            # Handle different export formats
            path_obj = Path(file_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            if export_format == ExportFormat.CSV:
                self._data_session.df.to_csv(path_obj, index=False, encoding=encoding)
            elif export_format == ExportFormat.TSV:
                self._data_session.df.to_csv(path_obj, sep="\t", index=False, encoding=encoding)
            elif export_format == ExportFormat.JSON:
                self._data_session.df.to_json(path_obj, orient="records", indent=2)
            elif export_format == ExportFormat.EXCEL:
                self._data_session.df.to_excel(path_obj, index=False)
            elif export_format == ExportFormat.PARQUET:
                self._data_session.df.to_parquet(path_obj, index=False)
            else:
                return {"success": False, "error": f"Unsupported format: {export_format}"}

            return {
                "success": True,
                "file_path": str(path_obj),
                "rows": len(self._data_session.df),
                "columns": len(self._data_session.df.columns),
            }
        except (OSError, PermissionError, ValueError, TypeError, UnicodeError) as e:
            return {"success": False, "error": str(e)}

    def rollback(self, steps: int = 1) -> bool:
        """Rollback operations by specified number of steps."""
        if self._data_session.original_df is None:
            return False

        if steps >= len(self.operations_history):
            # Rollback to original state
            self._data_session.df = self._data_session.original_df.copy()
            self.operations_history = []
            return True

        # This is a simplified rollback - in production, you'd replay operations
        logger.warning("Partial rollback not fully implemented")
        return False

    async def enable_auto_save(self, config: dict[str, Any]) -> AutoSaveOperationResult:
        """Enable or update auto-save configuration."""
        try:
            # Update configuration
            self.auto_save_config = AutoSaveConfig.from_dict(config)
            self.auto_save_manager = AutoSaveManager(
                self.session_id,
                self.auto_save_config,
                self._data_session.file_path,  # Pass the original file path
            )

            # Start periodic save if needed
            if self.auto_save_config.enabled:
                await self.auto_save_manager.start_periodic_save(self._save_callback)

            return {
                "success": True,
                "message": "Auto-save configuration updated",
                "config": self.auto_save_config.to_dict(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def disable_auto_save(self) -> AutoSaveOperationResult:
        """Disable auto-save."""
        try:
            await self.auto_save_manager.stop_periodic_save()
            self.auto_save_config.enabled = False
            return {"success": True, "message": "Auto-save disabled"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_auto_save_status(self) -> AutoSaveOperationResult:
        """Get current auto-save status."""
        return self.auto_save_manager.get_status()

    async def manual_save(self) -> AutoSaveOperationResult:
        """Manually trigger a save."""
        return await self.auto_save_manager.trigger_save(self._save_callback, "manual")

    async def undo(self) -> UndoRedoOperationResult:
        """Undo the last operation."""
        if not self.history_manager:
            raise HistoryNotEnabledError(self.session_id)

        if not self.history_manager.can_undo():
            return {"success": False, "error": "No operations to undo"}

        try:
            operation, data_snapshot = self.history_manager.undo()

            if data_snapshot is not None and operation is not None:
                self._data_session.df = data_snapshot

                # Trigger auto-save if configured
                if self.auto_save_manager.should_save_after_operation():
                    await self.auto_save_manager.trigger_save(self._save_callback, "undo")

                return {
                    "success": True,
                    "message": f"Undid operation: {operation.operation_type}",
                    "operation": operation.to_dict(),
                    "can_undo": self.history_manager.can_undo(),
                    "can_redo": self.history_manager.can_redo(),
                }
            return {"success": False, "error": "No snapshot available for undo"}

        except HistoryNotEnabledError as e:
            logger.error("History operation failed: %s", e.message)
            return {"success": False, "error": e.to_dict()}
        except Exception as e:
            logger.error("Unexpected error during undo: %s", str(e))
            return {
                "success": False,
                "error": {"type": "UnexpectedError", "message": str(e)},
            }

    async def redo(self) -> UndoRedoOperationResult:
        """Redo the previously undone operation."""
        if not self.history_manager:
            raise HistoryNotEnabledError(self.session_id)

        if not self.history_manager.can_redo():
            return {"success": False, "error": "No operations to redo"}

        try:
            operation, data_snapshot = self.history_manager.redo()

            if data_snapshot is not None and operation is not None:
                self._data_session.df = data_snapshot

                # Trigger auto-save if configured
                if self.auto_save_manager.should_save_after_operation():
                    await self.auto_save_manager.trigger_save(self._save_callback, "redo")

                return {
                    "success": True,
                    "message": f"Redid operation: {operation.operation_type}",
                    "operation": operation.to_dict(),
                    "can_undo": self.history_manager.can_undo(),
                    "can_redo": self.history_manager.can_redo(),
                }
            return {"success": False, "error": "No snapshot available for redo"}

        except HistoryNotEnabledError as e:
            logger.error("History operation failed: %s", e.message)
            return {"success": False, "error": e.to_dict()}
        except Exception as e:
            logger.error("Unexpected error during redo: %s", str(e))
            return {
                "success": False,
                "error": {"type": "UnexpectedError", "message": str(e)},
            }

    def get_history(self, limit: int | None = None) -> HistoryResult:
        """Get operation history."""
        if not self.history_manager:
            # Return legacy history if new history is not enabled
            return {
                "success": True,
                "history": (self.operations_history[-limit:] if limit else self.operations_history),
                "total": len(self.operations_history),
            }

        try:
            history = self.history_manager.get_history(limit)
            stats = self.history_manager.get_statistics()

            return {"success": True, "history": history, "statistics": stats}
        except HistoryError as e:
            logger.error("History operation failed: %s", e.message)
            return {"success": False, "error": e.to_dict()}
        except Exception as e:
            logger.error("Unexpected error getting history: %s", str(e))
            return {
                "success": False,
                "error": {"type": "UnexpectedError", "message": str(e)},
            }

    async def restore_to_operation(self, operation_id: str) -> UndoRedoOperationResult:
        """Restore data to a specific operation point."""
        if not self.history_manager:
            raise HistoryNotEnabledError(self.session_id)

        try:
            data_snapshot = self.history_manager.restore_to_operation(operation_id)

            if data_snapshot is not None:
                self.df = data_snapshot

                # Trigger auto-save if configured
                if self.auto_save_manager.should_save_after_operation():
                    await self.auto_save_manager.trigger_save(self._save_callback, "restore")

                return {
                    "success": True,
                    "message": f"Restored to operation {operation_id}",
                    "shape": (
                        self._data_session.df.shape if self._data_session.df is not None else (0, 0)
                    ),
                }
            return {
                "success": False,
                "error": f"Could not restore to operation {operation_id}",
            }

        except HistoryNotEnabledError as e:
            logger.error("History operation failed: %s", e.message)
            return {"success": False, "error": e.to_dict()}
        except Exception as e:
            logger.error("Unexpected error during restore: %s", str(e))
            return {
                "success": False,
                "error": {"type": "UnexpectedError", "message": str(e)},
            }

    async def clear(self) -> None:
        """Clear session data to free memory."""
        # Stop auto-save if running
        await self.auto_save_manager.stop_periodic_save()

        # Clear history if enabled
        if self.history_manager:
            self.history_manager.clear_history()

        # Clear data session
        self._data_session.clear_data()
        self.operations_history.clear()


class SessionManager:
    """Manages multiple CSV sessions with lifecycle and cleanup."""

    # Implementation: Session manager with capacity limits and TTL management
    def __init__(self, max_sessions: int = 100, ttl_minutes: int = 60):
        """Initialize session manager with limits."""
        self.sessions: dict[str, CSVSession] = {}
        self.max_sessions = max_sessions
        self.ttl_minutes = ttl_minutes
        self.sessions_to_cleanup: set = set()

    def get_session(self, session_id: str) -> CSVSession | None:
        """Get a session by ID without creating it if it doesn't exist.

        Use this for read-only operations to avoid unwanted session creation side effects.

        Args:
            session_id: The session identifier

        Returns:
            The session if it exists, None otherwise
        """
        session = self.sessions.get(session_id)
        if session and not session.is_expired():
            session.update_access_time()
            return session
        return None

    def get_or_create_session(self, session_id: str) -> CSVSession:
        """Get a session by ID, creating it if it doesn't exist."""
        session = self.sessions.get(session_id)
        if not session:
            # Create new session inline
            self._cleanup_expired()

            if len(self.sessions) >= self.max_sessions:
                # Remove oldest session
                oldest = min(self.sessions.values(), key=lambda s: s.lifecycle.last_accessed)
                del self.sessions[oldest.session_id]

            session = CSVSession(session_id=session_id, ttl_minutes=self.ttl_minutes)
            self.sessions[session.session_id] = session
            logger.info("Created new session: %s", session.session_id)
        else:
            session.update_access_time()
        return session

    async def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        if session_id in self.sessions:
            await self.sessions[session_id].clear()
            del self.sessions[session_id]
            logger.info("Removed session: %s", session_id)
            return True
        return False

    def list_sessions(self) -> list[SessionInfo]:
        """List all active sessions."""
        self._cleanup_expired()
        return [session.get_info() for session in self.sessions.values() if session.has_data()]

    def _cleanup_expired(self) -> None:
        """Mark expired sessions for cleanup."""
        expired = [sid for sid, session in self.sessions.items() if session.is_expired()]
        self.sessions_to_cleanup.update(expired)
        if expired:
            logger.info("Marked %s expired sessions for cleanup", len(expired))

    async def cleanup_marked_sessions(self) -> None:
        """Clean up sessions marked for removal."""
        for session_id in list(self.sessions_to_cleanup):
            await self.remove_session(session_id)
            self.sessions_to_cleanup.discard(session_id)

    def export_session_history(self, session_id: str) -> SessionHistoryExport:
        """Export session history as JSON."""
        session = self.get_or_create_session(session_id)

        return SessionHistoryExport(
            session_id=session.session_id,
            created_at=session.lifecycle.created_at.isoformat(),
            operations=session.operations_history,
            metadata=session.metadata,
        )


# Global session manager instance
_session_manager: SessionManager | None = None


# Implementation: Singleton pattern for global session manager
def get_session_manager() -> SessionManager:
    """Get global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_or_create_session(session_id: str) -> CSVSession:
    """Get or create session with elegant interface.

    Provides dictionary-like access: session = get_or_create_session(session_id)
    Returns existing session or creates new empty session.

    Args:
        session_id: The session identifier

    Returns:
        CSVSession (existing or newly created)
    """
    manager = get_session_manager()
    session = manager.get_or_create_session(session_id)

    if not session:
        # Create new session with the specified ID
        session = CSVSession(session_id=session_id)
        manager.sessions[session_id] = session

    return session
