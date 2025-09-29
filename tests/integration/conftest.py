"""Fixtures for integration tests."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from fastmcp import Client


def get_fixture_path(fixture_name: str) -> str:
    """Convert a fixture name to an absolute real filesystem path.

    Args:
        fixture_name: Name of the fixture file (e.g., "sample.csv")

    Returns:
        Absolute real filesystem path as string

    Raises:
        ValueError: If fixture_name contains path separators or resolves outside fixtures directory

    Example:
        get_fixture_path("sample.csv") -> "/real/absolute/path/to/tests/fixtures/sample.csv"
    """
    # Security: Reject empty or whitespace-only names
    if not fixture_name or not fixture_name.strip():
        msg = "Fixture name cannot be empty or whitespace-only"
        raise ValueError(msg)

    # Security: Validate fixture name doesn't contain path separators
    if os.path.sep in fixture_name or (os.path.altsep and os.path.altsep in fixture_name):
        msg = f"Fixture name cannot contain path separators: {fixture_name}"
        raise ValueError(msg)

    # Security: Reject relative path components
    if ".." in fixture_name or fixture_name.startswith("."):
        msg = f"Fixture name cannot contain relative path components: {fixture_name}"
        raise ValueError(msg)

    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    fixture_path = fixtures_dir / fixture_name
    resolved_path = os.path.realpath(fixture_path)

    # Security: Ensure resolved path is within fixtures directory
    fixtures_real_path = os.path.realpath(fixtures_dir)
    if not resolved_path.startswith(fixtures_real_path + os.path.sep):
        msg = f"Resolved path outside fixtures directory: {resolved_path}"
        raise ValueError(msg)

    return resolved_path


@pytest.fixture
async def databeak_client() -> AsyncGenerator[Client, None]:
    """Pytest fixture providing a FastMCP Client connected to DataBeak server.

    This fixture creates a direct in-memory connection to the DataBeak MCP server
    instance, bypassing subprocess overhead for fast integration testing.

    Yields:
        Client: FastMCP Client instance connected to DataBeak server

    Example:
        @pytest.mark.asyncio
        async def test_something(databeak_client):
            tools = await databeak_client.list_tools()
            result = await databeak_client.call_tool("get_session_info", {})
            assert result.is_error is False
    """
    # Import the server instance
    from databeak.server import mcp

    # Create FastMCP Client with direct server connection
    async with Client(mcp) as client:
        yield client
