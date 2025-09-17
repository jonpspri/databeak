"""Pytest configuration for CSV Editor tests."""

import asyncio
import sys
from asyncio import AbstractEventLoop
from collections.abc import Generator
from pathlib import Path

import pytest

from src.databeak.servers.io_server import load_csv_from_content
from tests.test_mock_context import create_mock_context

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session", autouse=True)
def cleanup_history_files() -> Generator[None]:
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
def event_loop() -> Generator[AbstractEventLoop]:
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
    csv_content = """product,price,quantity
Laptop,999.99,10
Mouse,29.99,50
Keyboard,79.99,25"""

    ctx = create_mock_context()
    _result = await load_csv_from_content(ctx, csv_content)
    return ctx.session_id
