"""Mock Context for testing FastMCP Context state management."""

import uuid
from typing import Any


class MockContext:
    """Mock implementation of FastMCP Context for testing."""

    def __init__(self, session_id: str | None = None, session_data: dict[str, Any] | None = None):
        self._session_id = session_id or uuid.uuid4().hex
        self._session_data = dict(session_data or {})

    @property
    def session_id(self) -> str:
        """Return the session ID."""
        return self._session_id

    async def info(self, message: str) -> None:
        """Mock info logging method."""

    async def debug(self, message: str) -> None:
        """Mock debug logging method."""

    async def error(self, message: str) -> None:
        """Mock error logging method."""

    async def warn(self, message: str) -> None:
        """Mock warning logging method."""

    async def warning(self, message: str) -> None:
        """Mock warning logging method (alias for warn)."""

    async def report_progress(self, progress: float) -> None:
        """Mock progress reporting method."""


def create_mock_context(
    session_id: str | None = None,
    session_data: dict[str, Any] | None = None,
) -> MockContext:
    """Create a mock context with session data."""
    return MockContext(session_id=session_id, session_data=session_data)
