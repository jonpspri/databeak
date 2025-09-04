"""Tests for server functionality and instruction loading."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.databeak.models import get_session_manager
from src.databeak.server import _load_instructions, main, mcp
from src.databeak.tools.io_operations import load_csv_from_content


class TestServerInstructions:
    """Test server instruction loading functionality."""

    def test_load_instructions_success(self):
        """Test successful instruction loading."""
        instructions = _load_instructions()
        assert isinstance(instructions, str)
        assert len(instructions) > 0
        assert "CSV Editor MCP Server" in instructions

    def test_load_instructions_file_not_found(self):
        """Test instruction loading when file doesn't exist."""
        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.side_effect = FileNotFoundError("File not found")

            instructions = _load_instructions()
            assert isinstance(instructions, str)
            assert "Instructions file not available" in instructions

    def test_load_instructions_generic_error(self):
        """Test instruction loading with generic error."""
        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.side_effect = Exception("Generic error")

            instructions = _load_instructions()
            assert isinstance(instructions, str)
            assert "Error loading instructions" in instructions

    def test_server_initialization(self):
        """Test that server initializes properly."""
        # Test that mcp server exists and has expected attributes
        assert mcp is not None
        assert hasattr(mcp, "name")
        assert mcp.name == "CSV Editor"
        assert hasattr(mcp, "instructions")
        assert isinstance(mcp.instructions, str)

    def test_main_function_definition(self):
        """Test that main function is properly defined."""
        assert callable(main)
        # Note: We can't test argparse easily without mocking sys.argv


@pytest.mark.asyncio
class TestServerSessionHandling:
    """Test server session handling functionality."""

    async def test_session_creation_and_cleanup(self):
        """Test session lifecycle management."""
        session_manager = get_session_manager()

        # Create session
        session_id = session_manager.create_session()
        assert session_id is not None

        # Verify session exists
        session = session_manager.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id

        # Test cleanup
        removed = await session_manager.remove_session(session_id)
        assert removed is True

        # Verify session is gone
        session = session_manager.get_session(session_id)
        assert session is None

    async def test_session_with_data_operations(self):
        """Test session operations with data."""
        # Create session with data
        result = await load_csv_from_content("name,age\nJohn,30\nJane,25")
        session_id = result["session_id"]

        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        # Test session has data
        assert session is not None
        assert session.data_session.has_data()
        assert session.data_session.df is not None
        assert len(session.data_session.df) == 2


class TestServerConfiguration:
    """Test server configuration and setup."""

    async def test_server_tool_registration(self):
        """Test that tools are properly registered."""
        # Test that server has tools registered
        assert hasattr(mcp, "_tool_manager")
        # Tools should be registered (exact count may vary)
        tools = await mcp.get_tools()
        assert len(tools) > 0

    def test_instructions_content_structure(self):
        """Test instruction content has expected structure."""
        instructions = mcp.instructions
        assert "Core Philosophy" in instructions
        assert "Coordinate System" in instructions
        assert "Getting Started" in instructions

    def test_server_prompts_exist(self):
        """Test that server has prompt templates."""
        # Test basic server structure has prompts
        assert hasattr(mcp, "_prompt_manager")

    def test_server_resources_exist(self):
        """Test that server has resource endpoints."""
        # Test basic server structure has resources
        assert hasattr(mcp, "_resource_manager")
