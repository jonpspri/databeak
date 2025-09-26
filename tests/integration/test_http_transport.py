"""Test HTTP transport integration."""

import pytest
from mcp import types

from tests.integration.conftest import get_fixture_path, get_server_fixture


class TestHttpTransport:
    """Test HTTP transport functionality."""

    @pytest.mark.asyncio
    async def test_http_server_starts_and_lists_tools(self):
        """Test that HTTP server starts and can list tools."""
        async with get_server_fixture("http") as server:
            tools = await server.list_tools()
            assert isinstance(tools, list)
            assert len(tools) > 0

            # Check for some expected tools - tools should be Tool objects with name attribute
            tool_names = {tool.name for tool in tools}
            expected_tools = {"list_sessions", "load_csv", "get_session_info"}

            # At least some expected tools should be present
            assert expected_tools.intersection(tool_names), (
                f"Expected tools not found in {tool_names}"
            )

    @pytest.mark.asyncio
    async def test_http_csv_loading(self):
        """Test loading CSV via HTTP transport."""
        async with get_server_fixture("http") as server:
            # Get the real path to the fixture
            csv_path = get_fixture_path("sample_data.csv")

            # Load the CSV file
            result = await server.call_tool("load_csv", {"file_path": csv_path})

            # Should return a CallToolResult
            assert isinstance(result, types.CallToolResult)

            # Verify the result contains expected data
            assert result.isError is False

    @pytest.mark.asyncio
    async def test_http_multiple_tool_calls(self):
        """Test making multiple tool calls via HTTP transport."""
        async with get_server_fixture("http") as server:
            # Call 1: List tools
            tools = await server.list_tools()
            assert len(tools) > 0

            # Call 2: List sessions (should be empty)
            sessions_result = await server.call_tool("list_sessions", {})
            assert isinstance(sessions_result, types.CallToolResult)

            # Call 3: Load a CSV file
            csv_path = get_fixture_path("sales_data.csv")
            load_result = await server.call_tool("load_csv", {"file_path": csv_path})
            assert isinstance(load_result, types.CallToolResult)
            assert load_result.isError is False

    @pytest.mark.asyncio
    async def test_http_vs_stdio_consistency(self):
        """Test that HTTP and stdio transports return consistent results."""
        # Test with stdio
        async with get_server_fixture("stdio") as stdio_server:
            stdio_tools = await stdio_server.list_tools()
            stdio_sessions = await stdio_server.call_tool("list_sessions", {})

        # Test with HTTP
        async with get_server_fixture("http") as http_server:
            http_tools = await http_server.list_tools()
            http_sessions = await http_server.call_tool("list_sessions", {})

        # Should have same number of tools
        assert len(stdio_tools) == len(http_tools)

        # Tool names should be the same
        stdio_tool_names = {tool.name for tool in stdio_tools}
        http_tool_names = {tool.name for tool in http_tools}
        assert stdio_tool_names == http_tool_names

        # Sessions result should be consistent (both should be empty initially)
        assert type(stdio_sessions) is type(http_sessions)
        assert isinstance(stdio_sessions, types.CallToolResult)
        assert isinstance(http_sessions, types.CallToolResult)
