"""Session service abstraction for dependency injection patterns.

This module provides abstractions for session management to improve testability and reduce coupling
between server modules and session management implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

from .data_models import SessionInfo

if TYPE_CHECKING:
    import pandas as pd

    from databeak.core.session import DatabeakSession


class SessionManagerProtocol(Protocol):
    """Protocol defining the session manager interface."""

    def get_or_create_session(self, session_id: str) -> DatabeakSession:
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
        """Initialize with injected session manager."""
        self.session_manager = session_manager

    def get_session_and_data(self, session_id: str) -> tuple[DatabeakSession, pd.DataFrame]:
        """Get session and validate data is loaded.

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
        """Initialize factory with session manager."""
        self.session_manager = session_manager

    def create_service(self, service_class: type[SessionService]) -> SessionService:
        """Create a service instance with injected dependencies.

        Returns:
            Configured service instance

        """
        return service_class(self.session_manager)


# Convenience functions for backward compatibility with existing code
def get_default_session_service_factory() -> SessionServiceFactory:
    """Get the default session service factory using the global session manager."""
    # Lazy import to avoid circular dependency
    from databeak.core.session import get_session_manager

    return SessionServiceFactory(get_session_manager())
