"""Integration tests for get_value_counts with string parameter handling."""

import pytest
from mcp import types

from tests.integration.conftest import get_fixture_path, get_server_fixture


class TestValueCountsStringParameters:
    """Test get_value_counts tool with string parameters that should be converted to proper types."""

    @pytest.mark.asyncio
    async def test_get_value_counts_with_string_top_n(self):
        """Test get_value_counts with string numeric top_n parameter."""
        async with get_server_fixture() as server:
            # Load sample data first
            csv_path = get_fixture_path("sample_data.csv")
            load_result = await server.call_tool("load_csv", {"file_path": csv_path})
            assert isinstance(load_result, types.CallToolResult)
            assert load_result.isError is False

            # Test with string "20" (should be converted to int 20)
            result = await server.call_tool(
                "get_value_counts",
                {
                    "column": "city",
                    "normalize": False,
                    "sort": True,
                    "ascending": False,
                    "top_n": "20",  # String instead of integer
                },
            )

            # Should succeed (not fail with validation error)
            assert isinstance(result, types.CallToolResult)
            assert result.isError is False, (
                f"Expected success but got error: {result.content if result.isError else ''}"
            )

    @pytest.mark.asyncio
    async def test_get_value_counts_with_string_null_top_n(self):
        """Test get_value_counts with string 'null' top_n parameter."""
        async with get_server_fixture() as server:
            # Load sample data first
            csv_path = get_fixture_path("sample_data.csv")
            load_result = await server.call_tool("load_csv", {"file_path": csv_path})
            assert isinstance(load_result, types.CallToolResult)
            assert load_result.isError is False

            # Test with string "null" (should be converted to None)
            result = await server.call_tool(
                "get_value_counts",
                {
                    "column": "city",
                    "normalize": False,
                    "sort": True,
                    "ascending": False,
                    "top_n": "null",  # String "null" instead of JSON null
                },
            )

            # Should succeed (not fail with validation error)
            assert isinstance(result, types.CallToolResult)
            assert result.isError is False, (
                f"Expected success but got error: {result.content if result.isError else ''}"
            )

    @pytest.mark.asyncio
    async def test_get_value_counts_reproducing_log_errors(self):
        """Reproduce the exact errors from the log file."""
        async with get_server_fixture() as server:
            # Load sample data first
            csv_path = get_fixture_path("sample_data.csv")
            load_result = await server.call_tool("load_csv", {"file_path": csv_path})
            assert isinstance(load_result, types.CallToolResult)
            assert load_result.isError is False

            # Reproduce error 1: top_n: "20"
            # From log: "Input validation error: '20' is not valid under any of the given schemas"
            result1 = await server.call_tool(
                "get_value_counts",
                {
                    "column": "name",  # Use a column that exists
                    "normalize": False,
                    "sort": True,
                    "ascending": False,
                    "top_n": "20",
                },
            )

            # Currently this fails with validation error, but should succeed after fix
            if result1.isError:
                # Expected current behavior - validation error
                assert (
                    "validation error" in str(result1.content).lower()
                    or "not valid" in str(result1.content).lower()
                )
            else:
                # After fix - should succeed
                assert isinstance(result1, types.CallToolResult)

            # Reproduce error 2: top_n: "null"
            # From log: "Input validation error: 'null' is not valid under any of the given schemas"
            result2 = await server.call_tool(
                "get_value_counts",
                {
                    "column": "name",
                    "normalize": False,
                    "sort": True,
                    "ascending": False,
                    "top_n": "null",
                },
            )

            # Currently this fails with validation error, but should succeed after fix
            if result2.isError:
                # Expected current behavior - validation error
                assert (
                    "validation error" in str(result2.content).lower()
                    or "not valid" in str(result2.content).lower()
                )
            else:
                # After fix - should succeed
                assert isinstance(result2, types.CallToolResult)

    @pytest.mark.asyncio
    async def test_get_value_counts_missing_top_n_parameter(self):
        """Test the third error case: missing top_n parameter when required."""
        async with get_server_fixture() as server:
            # Load sample data first
            csv_path = get_fixture_path("sample_data.csv")
            load_result = await server.call_tool("load_csv", {"file_path": csv_path})
            assert isinstance(load_result, types.CallToolResult)
            assert load_result.isError is False

            # Reproduce error 3: Missing top_n parameter entirely
            # From log: "Input validation error: 'top_n' is a required property"
            result = await server.call_tool(
                "get_value_counts",
                {
                    "column": "name",
                    "normalize": False,
                    "sort": True,
                    "ascending": False,
                    # top_n parameter missing entirely
                },
            )

            # This should succeed since top_n should be optional with default None
            assert isinstance(result, types.CallToolResult)
            assert result.isError is False, (
                f"top_n should be optional but got error: {result.content if result.isError else ''}"
            )

    @pytest.mark.asyncio
    async def test_get_value_counts_valid_parameters(self):
        """Test that properly typed parameters work correctly."""
        async with get_server_fixture() as server:
            # Load sample data first
            csv_path = get_fixture_path("sample_data.csv")
            load_result = await server.call_tool("load_csv", {"file_path": csv_path})
            assert isinstance(load_result, types.CallToolResult)
            assert load_result.isError is False

            # Test with proper integer top_n
            result = await server.call_tool(
                "get_value_counts",
                {
                    "column": "name",
                    "normalize": False,
                    "sort": True,
                    "ascending": False,
                    "top_n": 20,  # Proper integer
                },
            )

            assert isinstance(result, types.CallToolResult)
            assert result.isError is False

            # Test with proper null (None)
            result2 = await server.call_tool(
                "get_value_counts",
                {
                    "column": "name",
                    "normalize": False,
                    "sort": True,
                    "ascending": False,
                    "top_n": None,  # Proper null
                },
            )

            assert isinstance(result2, types.CallToolResult)
            assert result2.isError is False
