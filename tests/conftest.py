"""Pytest configuration for CSV Editor tests."""

import asyncio
import os
import sys
import uuid
from asyncio import AbstractEventLoop
from collections.abc import Generator
from pathlib import Path

import pytest

from src.databeak.models import get_session_manager
from src.databeak.servers.io_server import load_csv_from_content
from tests.test_mock_context import create_mock_context

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session", autouse=True)
def _cleanup_history_files() -> Generator[None]:
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


# Isolation fixtures for parallel execution safety
@pytest.fixture
def isolated_session_id():
    """Provide unique session ID for each test to prevent interference."""
    return uuid.uuid4().hex


@pytest.fixture
def temp_work_dir(tmp_path):
    """Provide isolated temporary directory per test to avoid resource contention."""
    work_dir = tmp_path / "test_work"
    work_dir.mkdir()
    old_cwd = Path.cwd()
    os.chdir(work_dir)
    yield work_dir
    os.chdir(old_cwd)


@pytest.fixture
def isolated_context(isolated_session_id):
    """Provide isolated mock context with unique session ID."""
    return create_mock_context(isolated_session_id)


@pytest.fixture
async def isolated_session_with_cleanup(isolated_session_id, temp_work_dir):
    """Create isolated session with automatic cleanup to prevent test interference."""
    manager = get_session_manager()
    session = manager.get_or_create_session(isolated_session_id)
    yield session
    # Cleanup: Remove session after test completes
    try:
        await manager.remove_session(isolated_session_id)
    except Exception as e:
        # Log the exception instead of silently passing
        import logging

        logging.getLogger(__name__).warning(
            "Failed to cleanup test session %s: %s", isolated_session_id, e
        )


@pytest.fixture
async def csv_session_with_data(isolated_context, temp_work_dir):
    """Create isolated CSV session with test data loaded."""
    csv_content = """product,price,quantity,category
Laptop,999.99,10,Electronics
Mouse,29.99,50,Electronics
Keyboard,79.99,25,Electronics
Desk,299.99,5,Furniture
Chair,199.99,8,Furniture"""

    result = await load_csv_from_content(isolated_context, content=csv_content, delimiter=",")

    # Return context and result for test use
    return isolated_context, result
