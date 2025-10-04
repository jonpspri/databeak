"""Example test demonstrating FastMCP Client fixture usage."""

import pytest


@pytest.mark.asyncio
async def test_list_tools(databeak_client) -> None:
    """Test that we can list available tools using the FastMCP Client fixture."""
    tools = await databeak_client.list_tools()

    # Should have many tools available
    assert len(tools) > 10

    # Check for some essential tools
    tool_names = [tool.name for tool in tools]
    assert "get_session_info" in tool_names
    assert "load_csv_from_content" in tool_names
    assert "export_csv" in tool_names


@pytest.mark.asyncio
async def test_get_session_info(databeak_client) -> None:
    """Test calling get_session_info tool with no data loaded."""
    from fastmcp.exceptions import ToolError

    # Should raise ToolError when no data is loaded
    with pytest.raises(ToolError, match="No data loaded"):
        await databeak_client.call_tool("get_session_info", {})


@pytest.mark.asyncio
async def test_load_csv_workflow(databeak_client) -> None:
    """Test a complete workflow: load CSV data, check session info, export."""
    # Step 1: Load some CSV data
    csv_content = "name,age,city\nAlice,30,New York\nBob,25,Boston\nCharlie,35,Chicago"

    load_result = await databeak_client.call_tool("load_csv_from_content", {"content": csv_content})

    assert load_result.is_error is False

    # Step 2: Check session info shows data is loaded
    info_result = await databeak_client.call_tool("get_session_info", {})
    assert info_result.is_error is False

    info_text = info_result.content[0].text
    # The info should contain data about rows and columns in JSON format
    assert "row_count" in info_text or "3" in info_text
    assert "column_count" in info_text or "3" in info_text

    # Step 3: Export the data back
    export_result = await databeak_client.call_tool(
        "export_csv", {"file_path": "/tmp/test_export.csv", "index": False}
    )

    assert export_result.is_error is False
    exported_content = export_result.content[0].text
    # The export result contains status information about the export
    assert "success" in exported_content
    assert "rows_exported" in exported_content
    assert "3" in exported_content  # Should indicate 3 rows were exported


@pytest.mark.asyncio
async def test_session_isolation(databeak_client) -> None:
    """Test that sessions are properly cleaned up between tests."""
    from fastmcp.exceptions import ToolError

    # This test should start with no data (proving cleanup from previous test)
    # Should raise ToolError when no data is loaded
    with pytest.raises(ToolError, match="No data loaded"):
        await databeak_client.call_tool("get_session_info", {})
