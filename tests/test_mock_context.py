"""Mock Context for testing FastMCP Context state management."""

from typing import Any


class MockContext:
    """Mock implementation of FastMCP Context for testing."""

    def __init__(
        self, session_id: str = "test_session", session_data: dict[str, Any] | None = None
    ):
        self._session_id = session_id
        self._session_data = session_data or {}

    @property
    def session_id(self) -> str:
        """Return the session ID."""
        return self._session_id

    async def info(self, message: str) -> None:
        """Mock info logging method."""
        pass

    async def debug(self, message: str) -> None:
        """Mock debug logging method."""
        pass

    async def error(self, message: str) -> None:
        """Mock error logging method."""
        pass

    async def warn(self, message: str) -> None:
        """Mock warning logging method."""
        pass

    async def report_progress(self, progress: float) -> None:
        """Mock progress reporting method."""
        pass


def create_mock_context(session_id: str = "test_session") -> MockContext:
    """Create a basic mock context with a session ID."""
    return MockContext(session_id=session_id)


def create_mock_context_with_session_data(
    session_id: str = "test_session", session_data: dict[str, Any] | None = None
) -> MockContext:
    """Create a mock context with session data."""
    return MockContext(session_id=session_id, session_data=session_data)
