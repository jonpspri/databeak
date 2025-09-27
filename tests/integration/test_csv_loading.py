"""Integration tests for CSV loading functionality."""

from pathlib import Path

import pytest
from mcp import types

from tests.integration.conftest import get_fixture_path, get_server_fixture


class TestCsvLoading:
    """Test CSV file loading and basic operations."""

    @pytest.mark.asyncio
    async def test_load_sample_data(self):
        """Test loading a sample CSV file."""
        async with get_server_fixture() as server:
            # Get the real path to the fixture
            csv_path = get_fixture_path("sample_data.csv")

            # Load the CSV file
            result = await server.call_tool("load_csv", {"file_path": csv_path})

            # Should return a CallToolResult
            assert isinstance(result, types.CallToolResult)

            # Verify the result contains expected data
            assert result.isError is False

    @pytest.mark.asyncio
    async def test_load_sample_data_string_header(self):
        """Test loading a sample CSV file."""
        async with get_server_fixture() as server:
            # Get the real path to the fixture
            csv_path = get_fixture_path("sample_data.csv")

            # Load the CSV file
            result = await server.call_tool("load_csv", {"file_path": csv_path, "header": "0"})
            # result = await server.call_tool("load_csv", {"file_path": csv_path, "header": "0"})

            # Should return a CallToolResult
            assert isinstance(result, types.CallToolResult)

            # Verify the result contains expected data
            assert result.isError is False

    @pytest.mark.asyncio
    async def test_load_sales_data_and_get_info(self):
        """Test loading sales data and getting session info."""
        async with get_server_fixture() as server:
            # Load sales data
            csv_path = get_fixture_path("sales_data.csv")
            load_result = await server.call_tool("load_csv", {"file_path": csv_path})
            assert isinstance(load_result, types.CallToolResult)

            # Verify the load was successful
            assert load_result.isError is False

    @pytest.mark.asyncio
    async def test_load_missing_values_csv(self):
        """Test loading CSV with missing values."""
        async with get_server_fixture() as server:
            csv_path = get_fixture_path("missing_values.csv")

            result = await server.call_tool("load_csv", {"file_path": csv_path})
            assert isinstance(result, types.CallToolResult)

            # Verify the load was successful
            assert result.isError is False

    @pytest.mark.asyncio
    async def test_fixture_path_resolution(self):
        """Test that the fixture path helper works correctly."""
        csv_path = get_fixture_path("sample_data.csv")

        # Should be an absolute path
        assert Path(csv_path).is_absolute()

        # Should end with the fixture name
        assert csv_path.endswith("sample_data.csv")

        # Should contain tests/fixtures in the path
        assert "tests/fixtures" in csv_path
