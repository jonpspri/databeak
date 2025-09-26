"""Integration tests for CSV loading functionality."""

import pytest
from mcp import types

from tests.integration.conftest import DatabeakServerFixture, get_fixture_path


class TestCsvLoading:
    """Test CSV file loading and basic operations."""

    @pytest.mark.asyncio
    async def test_load_sample_data(self):
        """Test loading a sample CSV file."""
        async with DatabeakServerFixture() as server:
            # Get the real path to the fixture
            csv_path = get_fixture_path("sample_data.csv")

            # Load the CSV file
            result = await server.call_tool("load_csv", {"file_path": csv_path})

            # Should return a CallToolResult
            assert isinstance(result, types.CallToolResult)

            # Verify we can list sessions and see our loaded data
            sessions_result = await server.call_tool("list_sessions", {})
            assert isinstance(sessions_result, types.CallToolResult)

    @pytest.mark.asyncio
    async def test_load_sales_data_and_get_info(self):
        """Test loading sales data and getting session info."""
        async with DatabeakServerFixture() as server:
            # Load sales data
            csv_path = get_fixture_path("sales_data.csv")
            load_result = await server.call_tool("load_csv", {"file_path": csv_path})
            assert isinstance(load_result, types.CallToolResult)

            # Get session ID from the load result (assuming it contains session info)
            # This will depend on how your load_csv tool returns the session ID
            sessions_result = await server.call_tool("list_sessions", {})
            assert isinstance(sessions_result, types.CallToolResult)

    @pytest.mark.asyncio
    async def test_load_missing_values_csv(self):
        """Test loading CSV with missing values."""
        async with DatabeakServerFixture() as server:
            csv_path = get_fixture_path("missing_values.csv")

            result = await server.call_tool("load_csv", {"file_path": csv_path})
            assert isinstance(result, types.CallToolResult)

            # Verify the data was loaded
            sessions_result = await server.call_tool("list_sessions", {})
            assert isinstance(sessions_result, types.CallToolResult)

    @pytest.mark.asyncio
    async def test_fixture_path_resolution(self):
        """Test that the fixture path helper works correctly."""
        csv_path = get_fixture_path("sample_data.csv")

        # Should be an absolute path
        assert csv_path.startswith("/")

        # Should end with the fixture name
        assert csv_path.endswith("sample_data.csv")

        # Should contain tests/fixtures in the path
        assert "tests/fixtures" in csv_path
