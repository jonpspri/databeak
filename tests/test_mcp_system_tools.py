"""Tests for system MCP tools (health check, server info)."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.databeak.server import _load_instructions, mcp
from src.databeak.tools.mcp_system_tools import HealthResult, ServerInfoResult


class TestInstructionsLoading:
    """Test instructions loading functionality."""

    def test_load_instructions_success(self) -> None:
        """Test successful loading of instructions."""
        instructions = _load_instructions()

        # Verify instructions are loaded
        assert instructions is not None
        assert len(instructions) > 0
        assert isinstance(instructions, str)

        # Check for expected content
        assert "DataBeak MCP Server" in instructions
        assert "Core Philosophy" in instructions
        assert "Getting Started" in instructions
        assert "Key Features" in instructions

    def test_load_instructions_content_structure(self) -> None:
        """Test that loaded instructions have expected structure."""
        instructions = _load_instructions()

        # Check for main sections (updated for new AI-focused structure)
        expected_sections = [
            "# DataBeak MCP Server",
            "Coordinate System",
            "Core Philosophy",
            "Getting Started (AI Workflow)",
        ]

        for section in expected_sections:
            assert section in instructions

    def test_load_instructions_file_not_found(self) -> None:
        """Test handling of missing instructions file."""
        with patch("src.databeak.server.Path") as mock_path:
            # Create a mock path that raises FileNotFoundError on read_text
            mock_instructions_path = mock_path.return_value.parent.__truediv__.return_value
            mock_instructions_path.read_text.side_effect = FileNotFoundError("File not found")

            instructions = _load_instructions()
            # Should return fallback message
            assert "Instructions file not available" in instructions

    def test_load_instructions_generic_error(self) -> None:
        """Test handling of generic errors during instructions loading."""
        with patch("src.databeak.server.Path") as mock_path:
            # Create a mock path that raises a generic error
            mock_instructions_path = mock_path.return_value.parent.__truediv__.return_value
            mock_instructions_path.read_text.side_effect = Exception("Generic error")

            instructions = _load_instructions()
            # Should return fallback message
            assert "Error loading instructions" in instructions

    def test_load_instructions_with_custom_file(self) -> None:
        """Test loading instructions from a temporary file."""
        # Create a temporary instructions file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as temp_file:
            temp_content = "# Test DataBeak\n\nThis is a test instructions file."
            temp_file.write(temp_content)
            temp_file.flush()

            # Mock the path to point to our temp file
            with patch("src.databeak.server.Path") as mock_path:
                mock_instructions_path = mock_path.return_value.parent.__truediv__.return_value
                mock_instructions_path.read_text.return_value = temp_content

                instructions = _load_instructions()
                assert instructions == temp_content

            # Clean up
            Path(temp_file.name).unlink()


class TestServerInitialization:
    """Test DataBeak server initialization."""

    def test_server_has_correct_name(self) -> None:
        """Test that server is initialized with correct name."""
        assert mcp.name == "DataBeak"

    def test_server_has_instructions(self) -> None:
        """Test that server has instructions loaded."""
        # Server should have instructions
        assert mcp.instructions is not None
        assert len(mcp.instructions) > 0

    def test_server_instructions_content(self) -> None:
        """Test that server instructions contain expected content."""
        instructions = mcp.instructions

        # Check for key sections that should be in instructions
        expected_content = [
            "DataBeak MCP Server",
            "Core Philosophy",
        ]

        for content in expected_content:
            assert content in instructions

    def test_server_instructions_match_file(self) -> None:
        """Test that server instructions match the loaded file instructions."""
        server_instructions = mcp.instructions
        file_instructions = _load_instructions()

        assert server_instructions == file_instructions

    def test_server_instructions_not_empty_on_error(self) -> None:
        """Test that server instructions are never empty, even on error."""
        # Even if there's an error loading instructions, the server should have fallback content
        assert mcp.instructions is not None
        assert len(mcp.instructions) > 0


class TestServerTools:
    """Test server tool registration."""

    def test_server_has_health_check_tool(self) -> None:
        """Test that server has health check functionality."""
        # Import the tools to verify they exist in their modules
        from src.databeak.tools.mcp_system_tools import register_system_tools

        # Verify the registration function exists
        assert callable(register_system_tools)

    def test_server_is_fastmcp_instance(self) -> None:
        """Test that mcp server is a FastMCP instance."""
        # Verify the server is properly instantiated
        from fastmcp import FastMCP

        assert isinstance(mcp, FastMCP)


class TestSystemToolFunctions:
    """Test the actual system tool functions and Pydantic models."""

    @pytest.mark.asyncio
    async def test_health_check_result_structure(self) -> None:
        """Test health check returns properly structured HealthResult."""
        # Test that we can create a valid HealthResult
        result = HealthResult(
            status="healthy",
            version="1.0.0",
            active_sessions=2,
            max_sessions=10,
            session_ttl_minutes=60,
        )
        assert result.success is True  # Default value
        assert result.status == "healthy"
        assert result.active_sessions == 2

    @pytest.mark.asyncio
    async def test_server_info_result_structure(self) -> None:
        """Test server info returns properly structured ServerInfoResult."""
        result = ServerInfoResult(
            name="DataBeak",
            version="1.0.0",
            description="Test server",
            capabilities={"test": ["tool1", "tool2"]},
            supported_formats=["csv", "json"],
            max_file_size_mb=1024,
            session_timeout_minutes=60,
        )
        assert result.name == "DataBeak"
        assert "test" in result.capabilities
        assert len(result.supported_formats) == 2

    def test_pydantic_model_imports(self) -> None:
        """Test that Pydantic models can be imported and have correct structure."""
        # Test HealthResult model
        assert hasattr(HealthResult, "model_fields")
        assert "success" in HealthResult.model_fields
        assert "status" in HealthResult.model_fields
        assert "version" in HealthResult.model_fields
        assert "active_sessions" in HealthResult.model_fields

        # Test ServerInfoResult model
        assert hasattr(ServerInfoResult, "model_fields")
        assert "name" in ServerInfoResult.model_fields
        assert "capabilities" in ServerInfoResult.model_fields
        assert "supported_formats" in ServerInfoResult.model_fields

    def test_pydantic_model_validation(self) -> None:
        """Test Pydantic model validation catches errors."""
        # Test that invalid data raises validation errors
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            HealthResult(
                status="healthy",
                version="1.0.0",
                active_sessions="invalid",  # Should be int, not str
                max_sessions=10,
                session_ttl_minutes=60,
            )
