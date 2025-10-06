"""Integration tests for system server functionality via MCP client."""

import importlib.metadata

import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport


class TestSystemServerIntegration:
    """Test system server functionality through FastMCP client."""

    @pytest.mark.asyncio
    async def test_health_check_via_client(
        self, databeak_client: Client[FastMCPTransport]
    ) -> None:
        """Test health_check tool returns proper response via MCP client."""
        result = await databeak_client.call_tool("health_check", {})

        assert result.is_error is False
        content = result.content[0].text

        # Verify result contains expected fields
        assert "status" in content
        assert "version" in content
        assert "active_sessions" in content

    @pytest.mark.asyncio
    async def test_get_server_info_via_client(
        self, databeak_client: Client[FastMCPTransport]
    ) -> None:
        """Test get_server_info tool returns proper response via MCP client."""
        result = await databeak_client.call_tool("get_server_info", {})

        assert result.is_error is False
        content = result.content[0].text

        # Verify result contains expected fields
        assert "name" in content
        assert "DataBeak" in content
        assert "version" in content
        assert "capabilities" in content
        assert "supported_formats" in content

    @pytest.mark.asyncio
    async def test_get_server_info_returns_actual_version_via_client(
        self, databeak_client: Client[FastMCPTransport]
    ) -> None:
        """Test that get_server_info returns actual package version via MCP client.

        This is an integration test that verifies the version is correctly
        propagated through the entire MCP stack, not just the unit test level.
        """
        # Get the actual package version
        expected_version = importlib.metadata.version("databeak")

        # Call the tool via MCP client
        result = await databeak_client.call_tool("get_server_info", {})

        assert result.is_error is False
        content = result.content[0].text

        # Verify version is present and correct
        assert "version" in content
        assert expected_version in content
        # Verify it's not the fallback version
        assert "0.0.0" not in content
        # For v0.1.0 release
        assert "0.1.0" in content

    @pytest.mark.asyncio
    async def test_health_check_returns_version_via_client(
        self, databeak_client: Client[FastMCPTransport]
    ) -> None:
        """Test that health_check also returns correct version via MCP client."""
        # Get the actual package version
        expected_version = importlib.metadata.version("databeak")

        # Call the tool via MCP client
        result = await databeak_client.call_tool("health_check", {})

        assert result.is_error is False
        content = result.content[0].text

        # Verify version is present and correct
        assert "version" in content
        assert expected_version in content
        # Verify it's not the fallback version
        assert "0.0.0" not in content

    @pytest.mark.asyncio
    async def test_system_tools_available(
        self, databeak_client: Client[FastMCPTransport]
    ) -> None:
        """Test that system tools are available in the tool list."""
        tools = await databeak_client.list_tools()
        tool_names = {tool.name for tool in tools}

        # Verify system tools are registered
        assert "health_check" in tool_names
        assert "get_server_info" in tool_names
