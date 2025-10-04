"""Example integration test using direct FastMCP Client fixture."""

import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport


@pytest.mark.asyncio
async def test_direct_client_tool_listing(databeak_client: Client[FastMCPTransport]) -> None:
    """Test listing tools using direct FastMCP client."""
    tools = await databeak_client.list_tools()

    # Verify we get tools from all mounted servers
    tool_names = [tool.name for tool in tools]

    # Should have tools from multiple servers
    assert len(tools) > 10

    # Verify some key tools are present
    assert "get_session_info" in tool_names
    assert "load_csv_from_content" in tool_names
    assert "export_csv" in tool_names


@pytest.mark.asyncio
async def test_direct_client_tool_execution(databeak_client: Client[FastMCPTransport]) -> None:
    """Test executing tools using direct FastMCP client."""
    # Test loading CSV content first
    csv_content = "name,age\nAlice,30\nBob,25"
    load_result = await databeak_client.call_tool("load_csv_from_content", {"content": csv_content})

    assert load_result.is_error is False

    # Verify session now has data
    info_result = await databeak_client.call_tool("get_session_info", {})
    assert info_result.is_error is False
    assert info_result.content is not None


@pytest.mark.asyncio
async def test_direct_client_multiple_operations(databeak_client: Client[FastMCPTransport]) -> None:
    """Test multiple operations using direct client."""
    csv_content = "name,age\nAlice,30\nBob,25"

    # Test loading CSV content
    load_result = await databeak_client.call_tool("load_csv_from_content", {"content": csv_content})
    assert load_result.is_error is False

    # Test getting session info after loading
    info_result = await databeak_client.call_tool("get_session_info", {})
    assert info_result.is_error is False

    # Test listing tools again to ensure client is still responsive
    tools = await databeak_client.list_tools()
    assert len(tools) > 10

    # Verify key tools are still available
    tool_names = [tool.name for tool in tools]
    assert "get_session_info" in tool_names
    assert "load_csv_from_content" in tool_names
