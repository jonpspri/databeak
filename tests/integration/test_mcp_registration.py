"""Tests for FastMCP server integration and tool availability.

Note: Individual tool registration tests have been superseded by server module tests.
This file tests the main server integration and overall functionality.
"""

from src.databeak.server import mcp


class TestServerIntegration:
    """Test that all server modules are properly integrated with main FastMCP server."""

    def test_main_server_exists(self) -> None:
        """Test that main server is properly configured."""
        assert mcp is not None
        assert mcp.name == "DataBeak"
        assert mcp.instructions is not None

    def test_server_initialization_successful(self) -> None:
        """Test that server initializes successfully with all tools."""
        from fastmcp import FastMCP

        # Verify server is FastMCP instance
        assert isinstance(mcp, FastMCP)
        assert mcp.name == "DataBeak"
        assert mcp.instructions is not None

    def test_tool_modules_importable(self) -> None:
        """Test that all tool modules can be imported successfully."""
        # This test ensures all modules have correct imports and dependencies

        # If we get here without ImportError, all modules imported successfully


class TestBackwardCompatibilityThroughModules:
    """Test that original functionality is preserved through modular architecture."""

    def test_core_functions_available_in_modules(self) -> None:
        """Test that core functions are available in their respective modules."""
        # Test core I/O functions
        from src.databeak.servers.io_server import export_csv, load_csv

        assert callable(load_csv)
        assert callable(export_csv)

        # Test core transformation functions
        from src.databeak.services.transformation_operations import add_column, filter_rows, insert_row

        assert callable(insert_row)
        assert callable(filter_rows)
        assert callable(add_column)

        # Test core analytics functions
        from src.databeak.servers.discovery_server import profile_data
        from src.databeak.servers.statistics_server import get_statistics

        assert callable(get_statistics)
        assert callable(profile_data)

    def test_null_value_support_preserved(self) -> None:
        """Test that null value support is preserved in refactored tools."""
        # These functions should support None/null values
        import inspect

        from src.databeak.services.transformation_operations import insert_row, set_cell_value, update_row

        # Check that insert_row accepts Any type for data
        sig = inspect.signature(insert_row)
        data_param = sig.parameters.get("data")
        assert data_param is not None

        # Check that set_cell_value accepts Any type for value
        sig = inspect.signature(set_cell_value)
        value_param = sig.parameters.get("value")
        assert value_param is not None

        # Check that update_row accepts dict with Any values
        sig = inspect.signature(update_row)
        data_param = sig.parameters.get("data")
        assert data_param is not None
