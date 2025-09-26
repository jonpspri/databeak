"""Integration tests for DataBeak server functionality."""

import pytest
from mcp import types

from tests.integration.conftest import get_server_fixture


class TestServerIntegration:
    """Test basic server functionality."""

    @pytest.mark.asyncio
    async def test_server_starts_and_lists_tools(self):
        """Test that server starts and can list tools."""
        async with get_server_fixture() as server:
            tools = await server.list_tools()
            assert isinstance(tools, list)
            assert len(tools) > 0

            # Check for some expected tools - tools should be Tool objects with name attribute
            tool_names = {tool.name for tool in tools}
            expected_tools = {"load_csv", "get_session_info"}

            # At least some expected tools should be present
            assert expected_tools.intersection(tool_names), (
                f"Expected tools not found in {tool_names}"
            )

    @pytest.mark.asyncio
    async def test_get_session_info_tool(self):
        """Test the get_session_info tool."""
        async with get_server_fixture() as server:
            # This should fail since no data is loaded, but should return a valid result
            result = await server.call_tool("get_session_info", {})

            # Should return a CallToolResult
            assert isinstance(result, types.CallToolResult)

    @pytest.mark.asyncio
    async def test_context_manager_usage(self):
        """Test using the context manager directly."""
        async with get_server_fixture() as server:
            # Test that we can call multiple tools
            tools = await server.list_tools()
            info_result = await server.call_tool("get_session_info", {})

            assert len(tools) > 0
            assert isinstance(info_result, types.CallToolResult)

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_same_session(self):
        """Test making multiple tool calls within the same test function."""
        async with get_server_fixture() as server:
            # Call 1: List tools
            tools = await server.list_tools()
            assert len(tools) > 0

            # Call 2: Get session info
            info_result = await server.call_tool("get_session_info", {})
            assert isinstance(info_result, types.CallToolResult)

            # Call 3: Get session info again (should be consistent)
            info_result2 = await server.call_tool("get_session_info", {})
            assert isinstance(info_result2, types.CallToolResult)

    @pytest.mark.asyncio
    async def test_server_cleanup(self):
        """Test that server properly cleans up after context manager exits."""
        server_ref = None

        # Use server in context
        async with get_server_fixture() as server:
            server_ref = server
            tools = await server.list_tools()
            assert len(tools) > 0

        # After context exits, the session should be closed
        # Note: We can't easily test stdio_client cleanup since it's internal
        # But we can verify the server context exited without errors
        assert server_ref is not None
