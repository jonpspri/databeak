"""Fixtures for integration tests."""

from __future__ import annotations

import asyncio
import os
import random
import socket
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import httpx
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.message import SessionMessage

# Type alias for the HTTP client context manager
HttpClientContextManager = AbstractAsyncContextManager[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
        Callable[[], str | None],
    ]
]


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


def find_available_port(max_attempts: int = 10) -> int:
    """Find an available port by randomly selecting from high port range.

    Args:
        max_attempts: Maximum number of random ports to try

    Returns:
        Available port number

    Raises:
        RuntimeError: If no available port found after max_attempts
    """
    for _ in range(max_attempts):
        # Use ephemeral port range above 10000 to avoid system ports
        port = random.randint(10000, 65535)  # noqa: S311
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue  # Port in use, try another

    msg = f"Could not find available port after {max_attempts} attempts"
    raise RuntimeError(msg)


class DatabeakServerFixture(ABC):
    """Context manager for running a DataBeak server subprocess with stdio transport."""

    @abstractmethod
    async def _initialize_stream(self): ...

    async def __aenter__(self) -> DatabeakServerFixture:
        """Start the DataBeak server subprocess with stdio transport."""
        mcp_read, mcp_write = await self._initialize_stream()

        self.client_session = ClientSession(mcp_read, mcp_write)
        await self.client_session.__aenter__()
        await self.client_session.initialize()
        return self

    @abstractmethod
    async def _finalize_stream(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Clean up the client and stop the server."""
        await self.client_session.__aexit__(exc_type, exc_val, exc_tb)
        await self._finalize_stream(exc_type, exc_val, exc_tb)
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

    async def _finalize_stream(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._stdio_client.__aexit__(exc_type, exc_val, exc_tb)


class DatabeakHttpServerFixture(DatabeakServerFixture):
    """HTTP transport fixture for DataBeak server."""

    def __init__(self, host: str = "127.0.0.1") -> None:
        """Initialize HTTP server fixture.

        Args:
            host: Host to bind the server to
        """
        self.host = host
        self.port = find_available_port()
        self.process: asyncio.subprocess.Process | None = None
        self._http_client: HttpClientContextManager | None = None

    async def _initialize_stream(self):
        """Start HTTP server and create client connection."""
        # Start the DataBeak server process with HTTP transport
        await self._start_http_server()

        # Create HTTP client connection
        url = f"http://{self.host}:{self.port}/mcp"
        self._http_client = streamablehttp_client(url)

        # Get the read/write streams (ignoring session_id callback for now)
        stream_result = await self._http_client.__aenter__()
        mcp_read, mcp_write, _get_session_id = stream_result

        return mcp_read, mcp_write

    async def _start_http_server(self) -> None:
        """Start the DataBeak server subprocess with HTTP transport."""
        project_root = Path(__file__).parent.parent.parent

        # Use asyncio.create_subprocess_exec for async-safe subprocess creation
        self.process = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "databeak",
            "--transport",
            "http",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--log-level",
            "ERROR",  # Minimize server logs during testing
            cwd=project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=0,
        )

        # Wait for MCP server to be ready
        await self._wait_for_mcp_ready()

    async def _wait_for_mcp_ready(self, *, timeout: int = 30) -> None:  # noqa: ASYNC109
        """Wait for the MCP HTTP endpoint to be ready to accept connections."""
        import time

        start_time = time.time()
        mcp_url = f"http://{self.host}:{self.port}/mcp"

        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() - start_time < timeout:
                # Check if process has exited
                if self.process and self.process.returncode is not None:
                    stdout, stderr = await self.process.communicate()
                    msg = f"Server process exited with code {self.process.returncode}"
                    if stderr:
                        msg += f". Stderr: {stderr.decode()}"
                    raise RuntimeError(msg)

                # Test MCP endpoint readiness
                try:
                    # Try to make a simple HTTP request to the MCP endpoint
                    response = await client.get(mcp_url, headers={"Accept": "text/event-stream"})
                    if response.status_code in (200, 400, 405):  # Server responding
                        await asyncio.sleep(0.2)  # Give it a moment to fully initialize
                        return
                except (httpx.ConnectError, httpx.TimeoutException):
                    # Server not ready yet, continue waiting
                    pass
                except Exception:
                    # Other HTTP errors might indicate server is responding
                    await asyncio.sleep(0.2)
                    return

                await asyncio.sleep(0.1)

        # Timeout reached
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        msg = f"MCP endpoint failed to be ready within {timeout} seconds"
        raise RuntimeError(msg)

    async def _finalize_stream(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up HTTP client and server."""
        # Close HTTP client
        if hasattr(self, "_http_client") and self._http_client:
            try:
                await self._http_client.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:  # noqa: S110
                pass
            self._http_client = None

        # Stop server process
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
            except Exception:  # noqa: S110
                pass
            self.process = None


def get_server_fixture(transport: Literal["stdio", "http"] = "stdio") -> DatabeakServerFixture:
    if transport == "stdio":
        return DatabeakStdioServerFixture()
    if transport == "http":
        return DatabeakHttpServerFixture()
    msg = f"Invalid transport {transport}"
    raise ValueError(msg)
