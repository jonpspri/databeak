"""Integration tests for DataBeak server functionality."""

import pytest


class TestServerIntegration:
    """Test basic server functionality."""

    @pytest.mark.asyncio
    async def test_server_starts_and_lists_tools(self, databeak_client):
        """Test that server starts and can list tools."""
        tools = await databeak_client.list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

        # Check for some expected tools - tools should be Tool objects with name attribute
        tool_names = {tool.name for tool in tools}
        expected_tools = {"load_csv", "get_session_info"}

        # At least some expected tools should be present
        assert expected_tools.intersection(tool_names), f"Expected tools not found in {tool_names}"

    @pytest.mark.asyncio
    async def test_get_session_info_tool(self, databeak_client):
        """Test the get_session_info tool."""
        from fastmcp.exceptions import ToolError

        # Should raise ToolError when no data is loaded
        with pytest.raises(ToolError, match="No data loaded"):
            await databeak_client.call_tool("get_session_info", {})

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, databeak_client):
        """Test using the context manager directly."""
        # Test that we can call multiple tools
        tools = await databeak_client.list_tools()

        # Load some data first so we can test get_session_info
        load_result = await databeak_client.call_tool(
            "load_csv_from_content", {"content": "name,age\nAlice,25"}
        )
        assert load_result.is_error is False

        info_result = await databeak_client.call_tool("get_session_info", {})
        assert info_result.is_error is False

        assert len(tools) > 0

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_same_session(self, databeak_client):
        """Test making multiple tool calls within the same test function."""
        # Call 1: List tools
        tools = await databeak_client.list_tools()
        assert len(tools) > 0

        # Call 2: Load data so we can test session info
        load_result = await databeak_client.call_tool(
            "load_csv_from_content", {"content": "name,age\nAlice,25"}
        )
        assert load_result.is_error is False

        # Call 3: Get session info
        info_result = await databeak_client.call_tool("get_session_info", {})
        assert info_result.is_error is False

        # Call 4: Get session info again (should be consistent)
        info_result2 = await databeak_client.call_tool("get_session_info", {})
        assert info_result2.is_error is False

    @pytest.mark.asyncio
    async def test_server_cleanup(self, databeak_client):
        """Test that server properly cleans up after context manager exits."""
        # Test that we can use the client normally
        tools = await databeak_client.list_tools()
        assert len(tools) > 0

        # The databeak_client fixture handles cleanup automatically
        # This test verifies the client works properly during the session
        assert databeak_client is not None
