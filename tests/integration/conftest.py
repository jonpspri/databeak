"""Fixtures for integration tests."""

from __future__ import annotations

import os

from pathlib import Path
from typing import Any, Literal

from abc import ABC, abstractmethod

from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client


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

class DatabeakServerFixture(ABC):
    """Context manager for running a DataBeak server subprocess with stdio transport."""

    @abstractmethod
    async def _initialize_stream(self):
        ...

    async def __aenter__(self) -> DatabeakServerFixture:
        """Start the DataBeak server subprocess with stdio transport."""
        mcp_read, mcp_write = await self._initialize_stream()

        self.client_session = ClientSession(mcp_read, mcp_write)
        await self.client_session.__aenter__()
        await self.client_session.initialize()
        return self

    @abstractmethod
    async def _finalize_stream(self):
        ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Clean up the client and stop the server."""
        await self.client_session.__aexit__(None, None, None)
        await self._finalize_stream()
        return False

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> types.CallToolResult:
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

class DatabeakStdioServerFixture(DatabeakServerFixture):
    async def _initialize_stream(self):
        self._stdio_client = stdio_client(
            StdioServerParameters(command="uv", args=["run", "databeak"])
        )
        return await self._stdio_client.__aenter__()

    async def _finalize_stream(self):
        await self._stdio_client.__aexit__(None, None, None)

def get_server_fixture(transport: Literal["stdio", "http"] = "stdio"):

    if transport == "stdio":
        return DatabeakStdioServerFixture()
    # if transport == "http":
    #     return DatabeakHttpServerFixture()
    msg = "Invalid transport %s"
    raise ValueError(msg, transport)

