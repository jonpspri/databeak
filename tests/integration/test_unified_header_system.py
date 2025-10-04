"""Test unified header system across all CSV loading functions."""

import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport

from tests.integration.conftest import get_fixture_path


class TestUnifiedHeaderSystem:
    """Test that all CSV loading functions use consistent HeaderConfig system."""

    @pytest.mark.asyncio
    async def test_load_csv_all_header_modes(
        self, databeak_client: Client[FastMCPTransport]
    ) -> None:
        """Test load_csv with all header configuration modes."""
        csv_path = get_fixture_path("sample_data.csv")

        # Test auto-detect mode
        auto_result = await databeak_client.call_tool(
            "load_csv", {"file_path": csv_path, "header_config": {"mode": "auto"}}
        )
        assert auto_result.is_error is False

        # Test explicit row mode
        row_result = await databeak_client.call_tool(
            "load_csv",
            {"file_path": csv_path, "header_config": {"mode": "row", "row_number": 0}},
        )
        assert row_result.is_error is False

        # Test no header mode
        none_result = await databeak_client.call_tool(
            "load_csv", {"file_path": csv_path, "header_config": {"mode": "none"}}
        )
        assert none_result.is_error is False

    @pytest.mark.asyncio
    async def test_load_csv_from_content_all_header_modes(
        self, databeak_client: Client[FastMCPTransport]
    ) -> None:
        """Test load_csv_from_content with all header configuration modes."""
        content = "name,age,city\nAlice,25,NYC\nBob,30,LA"

        # Test auto-detect mode (default)
        auto_result = await databeak_client.call_tool(
            "load_csv_from_content", {"content": content, "header_config": {"mode": "auto"}}
        )
        assert auto_result.is_error is False

        # Test explicit row mode
        row_result = await databeak_client.call_tool(
            "load_csv_from_content",
            {"content": content, "header_config": {"mode": "row", "row_number": 0}},
        )
        assert row_result.is_error is False

        # Test no header mode
        none_result = await databeak_client.call_tool(
            "load_csv_from_content", {"content": content, "header_config": {"mode": "none"}}
        )
        assert none_result.is_error is False

    @pytest.mark.asyncio
    async def test_header_consistency_across_functions(
        self, databeak_client: Client[FastMCPTransport]
    ) -> None:
        """Test that all CSV loading functions handle headers consistently."""
        csv_path = get_fixture_path("sample_data.csv")
        content = "name,age,city\nAlice,25,NYC\nBob,30,LA"

        # Test that all three functions accept the same header_config format
        header_configs = [{"mode": "auto"}, {"mode": "row", "row_number": 0}, {"mode": "none"}]

        for config in header_configs:
            # load_csv
            file_result = await databeak_client.call_tool(
                "load_csv", {"file_path": csv_path, "header_config": config}
            )
            assert file_result.is_error is False

            # load_csv_from_content
            content_result = await databeak_client.call_tool(
                "load_csv_from_content", {"content": content, "header_config": config}
            )
            assert content_result.is_error is False

    @pytest.mark.asyncio
    async def test_default_header_behavior_consistency(
        self, databeak_client: Client[FastMCPTransport]
    ) -> None:
        """Test that default header behavior is consistent across all functions."""
        csv_path = get_fixture_path("sample_data.csv")
        content = "name,age,city\nAlice,25,NYC\nBob,30,LA"

        # Test default behavior (should all use auto-detect)
        file_result = await databeak_client.call_tool("load_csv", {"file_path": csv_path})
        content_result = await databeak_client.call_tool(
            "load_csv_from_content", {"content": content}
        )

        # Both should succeed with default auto-detect behavior
        assert file_result.is_error is False
        assert content_result.is_error is False

    @pytest.mark.asyncio
    async def test_header_mode_validation(self, databeak_client: Client[FastMCPTransport]) -> None:
        """Test that invalid header configurations are properly rejected."""
        csv_path = get_fixture_path("sample_data.csv")

        # Test invalid mode
        with pytest.raises(Exception, match="invalid|validation|error"):
            await databeak_client.call_tool(
                "load_csv", {"file_path": csv_path, "header_config": {"mode": "invalid"}}
            )

        # Test missing row_number for explicit row mode
        with pytest.raises(Exception, match="row_number|required|validation"):
            await databeak_client.call_tool(
                "load_csv",
                {
                    "file_path": csv_path,
                    "header_config": {"mode": "row"},  # Missing row_number
                },
            )
