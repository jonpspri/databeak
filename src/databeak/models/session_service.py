"""Session service abstraction for dependency injection patterns.

This module provides abstractions for session management to improve testability and reduce coupling
between server modules and session management implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC
from typing import TYPE_CHECKING, Protocol

from .data_models import SessionInfo

if TYPE_CHECKING:
    import pandas as pd

    from .csv_session import CSVSession


class SessionManagerProtocol(Protocol):
    """Protocol defining the session manager interface."""

    def get_or_create_session(self, session_id: str) -> CSVSession:
        """Get or create a session by ID."""
        ...

    def list_sessions(self) -> list[SessionInfo]:
        """List all active sessions."""
        ...

    async def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        ...


class SessionService(ABC):
    """Abstract base service for operations that require session management.

    This class provides a foundation for implementing server modules with dependency injection for
    session management, improving testability and reducing coupling.
    """

    def __init__(self, session_manager: SessionManagerProtocol) -> None:
        """Initialize with injected session manager.

        Args:
            session_manager: The session manager implementation to use
        """
        self.session_manager = session_manager

    def get_session_and_data(self, session_id: str) -> tuple[CSVSession, pd.DataFrame]:
        """Get session and validate data is loaded.

        Args:
            session_id: The session ID to retrieve

        Returns:
            Tuple of (session, dataframe)

        Raises:
            ValueError: If session not found or no data loaded
        """
        session = self.session_manager.get_or_create_session(session_id)

        if not session:
            msg = f"Session not found: {session_id}"
            raise ValueError(msg)

        if not session.has_data():
            msg = f"No data loaded in session: {session_id}"
            raise ValueError(msg)

        df = session.df
        if df is None:
            msg = f"Invalid data state in session: {session_id}"
            raise ValueError(msg)

        return session, df

    def validate_columns_exist(self, df: pd.DataFrame, columns: list[str]) -> None:
        """Validate that columns exist in the dataframe.

        Args:
            df: The dataframe to check
            columns: List of column names to validate

        Raises:
            ValueError: If any columns are missing
        """
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            msg = f"Columns not found: {missing_cols}"
            raise ValueError(msg)

    @abstractmethod
    def get_service_name(self) -> str:
        """Get the name of this service for logging/identification."""
        ...


class SessionServiceFactory:
    """Factory for creating session services with proper dependency injection."""

    def __init__(self, session_manager: SessionManagerProtocol) -> None:
        """Initialize factory with session manager.

        Args:
            session_manager: The session manager to inject into services
        """
        self.session_manager = session_manager

    def create_service(self, service_class: type[SessionService]) -> SessionService:
        """Create a service instance with injected dependencies.

        Args:
            service_class: The service class to instantiate

        Returns:
            Configured service instance
        """
        return service_class(self.session_manager)


# Convenience functions for backward compatibility with existing code
def get_default_session_service_factory() -> SessionServiceFactory:
    """Get the default session service factory using the global session manager."""
    from .csv_session import get_session_manager

    return SessionServiceFactory(get_session_manager())


class MockSessionManager:
    """Mock session manager for testing.

    This provides a simple in-memory implementation suitable for unit testing without requiring the
    full session management infrastructure.
    """

    def __init__(self) -> None:
        """Initialize empty mock session manager."""
        self.sessions: dict[str, CSVSession] = {}
        self.next_id = 1

    def get_or_create_session(self, session_id: str) -> CSVSession:
        """Get or create a session by ID."""
        session = self.sessions.get(session_id)
        if not session:
            # Create new session like the real implementation
            from .csv_session import CSVSession

            session = CSVSession(session_id=session_id)
            self.sessions[session_id] = session
        return session

    def list_sessions(self) -> list[SessionInfo]:
        """List all active sessions."""
        from datetime import datetime

        results = []
        for session_id, session in self.sessions.items():
            df = session.df if session.has_data() else None
            info = SessionInfo(
                session_id=session_id,
                created_at=datetime.now(UTC),
                last_accessed=datetime.now(UTC),
                row_count=len(df) if df is not None else 0,
                column_count=len(df.columns) if df is not None else 0,
                columns=list(df.columns) if df is not None else [],
                memory_usage_mb=0.0,
                operations_count=0,
                file_path=None,
            )
            results.append(info)
        return results

    async def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def add_test_data(self, session_id: str, df: pd.DataFrame) -> None:
        """Add test data to a session (for testing purposes)."""
        session = self.get_or_create_session(session_id)
        session.load_data(df, None)
