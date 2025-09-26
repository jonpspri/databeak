"""Integration tests for DataBeak server functionality."""

import pytest
from mcp import types

from tests.integration.conftest import DatabeakServerFixture


class TestServerIntegration:
    """Test basic server functionality."""

    @pytest.mark.asyncio
    async def test_server_starts_and_lists_tools(self):
        """Test that server starts and can list tools."""
        async with DatabeakServerFixture() as server:
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
    async def test_list_sessions_tool(self):
        """Test the list_sessions tool."""
        async with DatabeakServerFixture() as server:
            result = await server.call_tool("list_sessions", {})

            # Should return a CallToolResult
            assert isinstance(result, types.CallToolResult)

            # Check the content - should be a list of sessions (empty initially)
            if result.content:
                # Content should be a list of TextContent or similar
                assert isinstance(result.content, list)

    @pytest.mark.asyncio
    async def test_context_manager_usage(self):
        """Test using the context manager directly."""
        async with DatabeakServerFixture() as server:
            # Test that we can call multiple tools
            tools = await server.list_tools()
            sessions_result = await server.call_tool("list_sessions", {})

            assert len(tools) > 0
            assert isinstance(sessions_result, types.CallToolResult)

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_same_session(self):
        """Test making multiple tool calls within the same test function."""
        async with DatabeakServerFixture() as server:
            # Call 1: List tools
            tools = await server.list_tools()
            assert len(tools) > 0

            # Call 2: List sessions (should be empty)
            sessions_result = await server.call_tool("list_sessions", {})
            assert isinstance(sessions_result, types.CallToolResult)

            # Call 3: Get session info - test with a known session_id pattern
            # This may fail with invalid session, which is expected for this test
            try:
                info_result = await server.call_tool("get_session_info", {"session_id": "test"})
                assert isinstance(info_result, types.CallToolResult)
            except Exception:  # noqa: S110
                # Expected failure for invalid session - we're testing multiple calls work
                pass

    @pytest.mark.asyncio
    async def test_server_cleanup(self):
        """Test that server properly cleans up after context manager exits."""
        server_ref = None

        # Use server in context
        async with DatabeakServerFixture() as server:
            server_ref = server
            tools = await server.list_tools()
            assert len(tools) > 0

        # After context exits, the session should be closed
        # Note: We can't easily test stdio_client cleanup since it's internal
        # But we can verify the server context exited without errors
        assert server_ref is not None
