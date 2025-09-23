"""Session management for DataBeak CSV operations."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..exceptions import SessionExpiredError
from ..models.data_models import ExportFormat, SessionInfo
from ..models.data_session import DataSession

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class SessionLifecycle:
    """Manages session TTL and expiration logic."""

    def __init__(self, session_id: str, ttl_minutes: int = 60):
        """Initialize session lifecycle manager."""
        self.session_id = session_id
        self.ttl = timedelta(minutes=ttl_minutes)
        self.created_at = datetime.now(UTC)
        self.last_accessed = datetime.now(UTC)

    def update_access_time(self) -> None:
        """Update the last accessed time."""
        self.last_accessed = datetime.now(UTC)

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now(UTC) - self.last_accessed > self.ttl

    def validate_not_expired(self) -> None:
        """Raise exception if session is expired."""
        if self.is_expired():
            raise SessionExpiredError(self.session_id)

    def get_remaining_ttl(self) -> timedelta:
        """Get remaining time until expiration."""
        elapsed = datetime.now(UTC) - self.last_accessed
        return max(timedelta(0), self.ttl - elapsed)

    def extend_ttl(self, additional_minutes: int) -> None:
        """Extend the session TTL."""
        self.ttl += timedelta(minutes=additional_minutes)
        logger.info(
            "Extended TTL for session %s by %s minutes", self.session_id, additional_minutes
        )

    def get_lifecycle_info(self) -> dict[str, Any]:
        """Get session lifecycle information."""
        remaining = self.get_remaining_ttl()

        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "ttl_minutes": int(self.ttl.total_seconds() / 60),
            "remaining_minutes": int(remaining.total_seconds() / 60),
            "is_expired": self.is_expired(),
        }


class DatabeakSession:
    """CSV editing session for DataBeak operations."""

    def __init__(
        self,
        session_id: str | None = None,
        ttl_minutes: int = 60,
    ):
        """Initialize CSV session."""
        self.session_id = session_id or str(uuid4())

        # Core components
        self._data_session = DataSession(self.session_id)
        self.lifecycle = SessionLifecycle(self.session_id, ttl_minutes)

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
            operations_count=0,  # No longer tracking operations (simplified architecture)
            file_path=data_info["file_path"],
        )

    async def _save_callback(
        self,
        file_path: str,
        export_format: ExportFormat,
        encoding: str,
    ) -> dict[str, Any]:
        """Handle auto-save operations."""
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

    async def clear(self) -> None:
        """Clear session data to free memory."""
        # Clear data session
        self._data_session.clear_data()


class SessionManager:
    """Manages multiple CSV sessions with lifecycle and cleanup."""

    # Implementation: Session manager with capacity limits and TTL management
    def __init__(self, max_sessions: int = 100, ttl_minutes: int = 60):
        """Initialize session manager with limits."""
        self.sessions: dict[str, DatabeakSession] = {}
        self.max_sessions = max_sessions
        self.ttl_minutes = ttl_minutes
        self.sessions_to_cleanup: set = set()

    def get_session(self, session_id: str) -> DatabeakSession | None:
        """Get a session by ID without creating it if it doesn't exist.

        Use this for read-only operations to avoid unwanted session creation side effects.

        Returns:
            The session if it exists, None otherwise

        """
        session = self.sessions.get(session_id)
        if session and not session.is_expired():
            session.update_access_time()
            return session
        return None

    def get_or_create_session(self, session_id: str) -> DatabeakSession:
        """Get a session by ID, creating it if it doesn't exist."""
        session = self.sessions.get(session_id)
        if not session:
            # Create new session inline
            self._cleanup_expired()

            if len(self.sessions) >= self.max_sessions:
                # Remove oldest session
                oldest = min(self.sessions.values(), key=lambda s: s.lifecycle.last_accessed)
                del self.sessions[oldest.session_id]

            session = DatabeakSession(session_id=session_id, ttl_minutes=self.ttl_minutes)
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


# Global session manager instance
_session_manager = SessionManager()


# Implementation: Singleton pattern for global session manager
def get_session_manager() -> SessionManager:
    """Return the global Session Manager object."""
    return _session_manager


def get_or_create_session(session_id: str) -> DatabeakSession:
    """Get or create session with elegant interface.

    Provides dictionary-like access: session = get_or_create_session(session_id)
    Returns existing session or creates new empty session.

    Returns:
        DatabeakSession (existing or newly created)

    """
    manager = get_session_manager()
    session = manager.get_or_create_session(session_id)

    if not session:
        # Create new session with the specified ID
        session = DatabeakSession(session_id=session_id)
        manager.sessions[session_id] = session

    return session
