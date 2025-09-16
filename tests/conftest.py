"""Pytest configuration for CSV Editor tests."""

import asyncio
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session", autouse=True)
def cleanup_history_files() -> None:
    """Clean up history files created during testing."""
    yield  # Let all tests run first

    # Clean up any history files created during testing
    project_root = Path(__file__).parent.parent
    for history_file in project_root.glob("history_*.json"):
        try:
            history_file.unlink()
        except (OSError, FileNotFoundError):
            pass  # File might already be removed


@pytest.fixture(scope="session")
def event_loop() -> None:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_csv_data() -> str:
    """Provide sample CSV data for testing."""
    return """name,age,salary,department
Alice,30,60000,Engineering
Bob,25,50000,Marketing
Charlie,35,70000,Engineering
Diana,28,55000,Sales"""


@pytest.fixture
async def test_session() -> str:
    """Create a test session."""
    from src.databeak.models import get_session_manager
    from src.databeak.servers.io_server import load_csv_from_content
    from tests.test_mock_context import create_mock_context

    # Create session with sample data
    result = await load_csv_from_content(
        create_mock_context(),
        content="""product,price,quantity
Laptop,999.99,10
Mouse,29.99,50
Keyboard,79.99,25""",
        delimiter=",",
    )

    # Get the session ID from the context instead of result
    # Since LoadResult no longer contains session_id, we need to get it from the session manager
    manager = get_session_manager()
    sessions = manager.list_sessions()
    session_id = sessions[-1].session_id if sessions else "test-session-fallback"

    yield session_id

    # Cleanup
    await manager.remove_session(session_id)
