"""Fixtures for integration tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client

if TYPE_CHECKING:
    pass


def get_fixture_path(fixture_name: str) -> str:
    """Convert a fixture name to an absolute real filesystem path.

    Args:
        fixture_name: Name of the fixture file (e.g., "sample.csv")

    Returns:
        Absolute real filesystem path as string

    Example:
        get_fixture_path("sample.csv") -> "/real/absolute/path/to/tests/fixtures/sample.csv"
    """
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    fixture_path = fixtures_dir / fixture_name
    return os.path.realpath(fixture_path)


class DatabeakServerFixture:
    """Context manager for running a DataBeak server subprocess with stdio transport."""

    def __init__(self) -> None:
        """Initialize the server fixture."""

    async def __aenter__(self) -> DatabeakServerFixture:
        """Start the DataBeak server subprocess with stdio transport."""
        self.stdio_client = stdio_client(
            StdioServerParameters(command="uv", args=["run", "databeak"])
        )
        mcp_read, mcp_write = await self.stdio_client.__aenter__()  # First yield is the session

        self.client_session = ClientSession(mcp_read, mcp_write)
        await self.client_session.__aenter__()
        await self.client_session.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up the client and stop the server."""
        await self.client_session.__aexit__(None, None, None)
        await self.stdio_client.__aexit__(None, None, None)

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Call a tool on the server.

        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool parameters

        Returns:
            Tool response

        Raises:
            RuntimeError: If client is not initialized

        """
        if not self.client_session:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        return await self.client_session.call_tool(tool_name, args)

    async def list_tools(self) -> list[types.Tool]:
        """List available tools.

        Returns:
            List of tool information

        Raises:
            RuntimeError: If client is not initialized

        """
        if not self.client_session:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        result = await self.client_session.list_tools()
        return result.tools
