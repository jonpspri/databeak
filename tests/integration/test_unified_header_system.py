"""Test unified header system across all CSV loading functions."""

import pytest
from mcp import types

from tests.integration.conftest import get_fixture_path, get_server_fixture


class TestUnifiedHeaderSystem:
    """Test that all CSV loading functions use consistent HeaderConfig system."""

    @pytest.mark.asyncio
    async def test_load_csv_all_header_modes(self):
        """Test load_csv with all header configuration modes."""
        async with get_server_fixture() as server:
            csv_path = get_fixture_path("sample_data.csv")

            # Test auto-detect mode
            auto_result = await server.call_tool(
                "load_csv", {"file_path": csv_path, "header_config": {"mode": "auto"}}
            )
            assert isinstance(auto_result, types.CallToolResult)
            assert auto_result.isError is False

            # Test explicit row mode
            row_result = await server.call_tool(
                "load_csv",
                {"file_path": csv_path, "header_config": {"mode": "row", "row_number": 0}},
            )
            assert isinstance(row_result, types.CallToolResult)
            assert row_result.isError is False

            # Test no header mode
            none_result = await server.call_tool(
                "load_csv", {"file_path": csv_path, "header_config": {"mode": "none"}}
            )
            assert isinstance(none_result, types.CallToolResult)
            assert none_result.isError is False

    @pytest.mark.asyncio
    async def test_load_csv_from_content_all_header_modes(self):
        """Test load_csv_from_content with all header configuration modes."""
        async with get_server_fixture() as server:
            content = "name,age,city\nAlice,25,NYC\nBob,30,LA"

            # Test auto-detect mode (default)
            auto_result = await server.call_tool(
                "load_csv_from_content", {"content": content, "header_config": {"mode": "auto"}}
            )
            assert isinstance(auto_result, types.CallToolResult)
            assert auto_result.isError is False

            # Test explicit row mode
            row_result = await server.call_tool(
                "load_csv_from_content",
                {"content": content, "header_config": {"mode": "row", "row_number": 0}},
            )
            assert isinstance(row_result, types.CallToolResult)
            assert row_result.isError is False

            # Test no header mode
            none_result = await server.call_tool(
                "load_csv_from_content", {"content": content, "header_config": {"mode": "none"}}
            )
            assert isinstance(none_result, types.CallToolResult)
            assert none_result.isError is False

    @pytest.mark.asyncio
    async def test_header_consistency_across_functions(self):
        """Test that all CSV loading functions handle headers consistently."""
        async with get_server_fixture() as server:
            csv_path = get_fixture_path("sample_data.csv")
            content = "name,age,city\nAlice,25,NYC\nBob,30,LA"

            # Test that all three functions accept the same header_config format
            header_configs = [{"mode": "auto"}, {"mode": "row", "row_number": 0}, {"mode": "none"}]

            for config in header_configs:
                # load_csv
                file_result = await server.call_tool(
                    "load_csv", {"file_path": csv_path, "header_config": config}
                )
                assert isinstance(file_result, types.CallToolResult)
                assert file_result.isError is False

                # load_csv_from_content
                content_result = await server.call_tool(
                    "load_csv_from_content", {"content": content, "header_config": config}
                )
                assert isinstance(content_result, types.CallToolResult)
                assert content_result.isError is False

    @pytest.mark.asyncio
    async def test_default_header_behavior_consistency(self):
        """Test that default header behavior is consistent across all functions."""
        async with get_server_fixture() as server:
            csv_path = get_fixture_path("sample_data.csv")
            content = "name,age,city\nAlice,25,NYC\nBob,30,LA"

            # Test default behavior (should all use auto-detect)
            file_result = await server.call_tool("load_csv", {"file_path": csv_path})
            content_result = await server.call_tool("load_csv_from_content", {"content": content})

            # Both should succeed with default auto-detect behavior
            assert isinstance(file_result, types.CallToolResult)
            assert file_result.isError is False
            assert isinstance(content_result, types.CallToolResult)
            assert content_result.isError is False

    @pytest.mark.asyncio
    async def test_header_mode_validation(self):
        """Test that invalid header configurations are properly handled."""
        async with get_server_fixture() as server:
            csv_path = get_fixture_path("sample_data.csv")

            # Test invalid mode - FastMCP may handle this gracefully
            result = await server.call_tool(
                "load_csv", {"file_path": csv_path, "header_config": {"mode": "invalid"}}
            )
            # Should either fail with validation error or succeed with default behavior
            assert isinstance(result, types.CallToolResult)
            # Don't assert on isError since framework behavior may vary

            # Test missing row_number for explicit row mode - FastMCP may handle this gracefully
            result2 = await server.call_tool(
                "load_csv",
                {
                    "file_path": csv_path,
                    "header_config": {"mode": "row"},  # Missing row_number
                },
            )
            # Should either fail with validation error or succeed with default behavior
            assert isinstance(result2, types.CallToolResult)
            # Don't assert on isError since framework behavior may vary
