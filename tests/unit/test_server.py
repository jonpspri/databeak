"""Unit tests for server.py module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

from src.databeak.server import (
    _load_instructions,
    analyze_csv_prompt,
    data_cleaning_prompt,
    get_csv_cell,
    get_csv_data,
    get_csv_preview,
    get_csv_row,
    get_csv_schema,
    list_active_sessions,
    main,
)


class TestLoadInstructions:
    """Tests for _load_instructions function."""

    @patch("src.databeak.server.Path")
    def test_load_instructions_success(self, mock_path):
        """Test successful loading of instructions."""
        mock_file = MagicMock()
        mock_file.read_text.return_value = "Test instructions content"
        mock_path_instance = MagicMock()
        mock_path_instance.__truediv__.return_value = mock_file
        mock_path.return_value = mock_path_instance

        result = _load_instructions()

        assert result == "Test instructions content"
        mock_file.read_text.assert_called_once_with(encoding="utf-8")

    @patch("src.databeak.server.Path")
    @patch("src.databeak.server.logger")
    def test_load_instructions_file_not_found(self, mock_logger, mock_path):
        """Test instructions loading when file not found."""
        mock_file = MagicMock()
        mock_file.read_text.side_effect = FileNotFoundError()
        mock_path_instance = MagicMock()
        mock_path_instance.__truediv__.return_value = mock_file
        mock_path.return_value = mock_path_instance

        result = _load_instructions()

        assert "Instructions file not available" in result
        mock_logger.warning.assert_called_once()

    @patch("src.databeak.server.Path")
    @patch("src.databeak.server.logger")
    def test_load_instructions_other_error(self, mock_logger, mock_path):
        """Test instructions loading with other error."""
        mock_file = MagicMock()
        mock_file.read_text.side_effect = Exception("Permission denied")
        mock_path_instance = MagicMock()
        mock_path_instance.__truediv__.return_value = mock_file
        mock_path.return_value = mock_path_instance

        result = _load_instructions()

        assert "Error loading instructions" in result
        mock_logger.error.assert_called_once()


class TestResourceFunctions:
    """Tests for resource functions."""

    @pytest.fixture
    def mock_session_with_data(self):
        """Fixture providing mock session with DataFrame."""
        session = MagicMock()
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "city": ["NYC", "LA", "Chicago"],
            }
        )
        session.df = df
        session.has_data.return_value = True
        return session, df

    @pytest.fixture
    def mock_session_without_data(self):
        """Fixture providing mock session without data."""
        session = MagicMock()
        session.has_data.return_value = False
        return session

    @pytest.mark.asyncio
    @patch("src.databeak.server.get_session_manager")
    @patch("src.databeak.server.create_data_preview_with_indices")
    async def test_get_csv_data_success(
        self, mock_preview, mock_get_manager, mock_session_with_data
    ):
        """Test successful CSV data retrieval."""
        session, df = mock_session_with_data
        manager = MagicMock()
        manager.get_session.return_value = session
        mock_get_manager.return_value = manager
        mock_preview.return_value = {"preview": "data"}

        result = await get_csv_data("test-session")

        assert result["session_id"] == "test-session"
        assert result["shape"] == (3, 3)
        assert "preview" in result
        assert "columns_info" in result
        assert result["columns_info"]["columns"] == ["name", "age", "city"]
        mock_preview.assert_called_once_with(df, 10)

    @pytest.mark.asyncio
    @patch("src.databeak.server.get_session_manager")
    async def test_get_csv_data_session_not_found(self, mock_get_manager):
        """Test CSV data retrieval when session not found."""
        manager = MagicMock()
        manager.get_session.return_value = None
        mock_get_manager.return_value = manager

        result = await get_csv_data("invalid-session")

        assert "error" in result
        assert "Session not found" in result["error"]

    @pytest.mark.asyncio
    @patch("src.databeak.server.get_session_manager")
    async def test_get_csv_data_no_data(self, mock_get_manager, mock_session_without_data):
        """Test CSV data retrieval when no data loaded."""
        session = mock_session_without_data
        manager = MagicMock()
        manager.get_session.return_value = session
        mock_get_manager.return_value = manager

        result = await get_csv_data("no-data-session")

        assert "error" in result
        assert "no data loaded" in result["error"]

    @pytest.mark.asyncio
    @patch("src.databeak.server.get_session_manager")
    async def test_get_csv_schema_success(self, mock_get_manager, mock_session_with_data):
        """Test successful CSV schema retrieval."""
        session, df = mock_session_with_data
        manager = MagicMock()
        manager.get_session.return_value = session
        mock_get_manager.return_value = manager

        result = await get_csv_schema("test-session")

        assert result["session_id"] == "test-session"
        assert result["columns"] == ["name", "age", "city"]
        assert result["shape"] == (3, 3)
        assert "dtypes" in result

    @pytest.mark.asyncio
    @patch("src.databeak.server.get_session_manager")
    async def test_list_active_sessions(self, mock_get_manager):
        """Test listing active sessions."""
        manager = MagicMock()
        mock_session1 = MagicMock()
        mock_session2 = MagicMock()
        mock_session1.dict.return_value = {"id": "session1", "created": "2023-01-01"}
        mock_session2.dict.return_value = {"id": "session2", "created": "2023-01-02"}
        manager.list_sessions.return_value = [mock_session1, mock_session2]
        mock_get_manager.return_value = manager

        result = await list_active_sessions()

        assert len(result) == 2
        assert result[0]["id"] == "session1"
        assert result[1]["id"] == "session2"

    @pytest.mark.asyncio
    @patch("src.databeak.server._get_cell_value")
    async def test_get_csv_cell_success(self, mock_get_cell):
        """Test successful CSV cell retrieval."""
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "session_id": "test-session",
            "row_index": 0,
            "column": "name",
            "value": "Alice",
        }
        mock_get_cell.return_value = mock_result

        result = await get_csv_cell("test-session", "0", "name")

        assert result["session_id"] == "test-session"
        assert result["row_index"] == 0
        assert result["column"] == "name"
        assert result["value"] == "Alice"
        mock_get_cell.assert_called_once_with("test-session", 0, "name")

    @pytest.mark.asyncio
    @patch("src.databeak.server._get_cell_value")
    async def test_get_csv_cell_numeric_column(self, mock_get_cell):
        """Test CSV cell retrieval with numeric column index."""
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"value": "test"}
        mock_get_cell.return_value = mock_result

        result = await get_csv_cell("test-session", "1", "0")

        mock_get_cell.assert_called_once_with("test-session", 1, 0)

    @pytest.mark.asyncio
    async def test_get_csv_cell_invalid_row_index(self):
        """Test CSV cell retrieval with invalid row index."""
        result = await get_csv_cell("test-session", "invalid", "name")

        assert "error" in result
        assert "Invalid row index" in result["error"]

    @pytest.mark.asyncio
    @patch("src.databeak.server._get_cell_value")
    async def test_get_csv_cell_tool_error(self, mock_get_cell):
        """Test CSV cell retrieval with ToolError."""
        mock_get_cell.side_effect = ToolError("Cell not found")

        result = await get_csv_cell("test-session", "0", "name")

        assert "error" in result
        assert "Cell not found" in result["error"]

    @pytest.mark.asyncio
    @patch("src.databeak.server._get_row_data")
    async def test_get_csv_row_success(self, mock_get_row):
        """Test successful CSV row retrieval."""
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "session_id": "test-session",
            "row_index": 1,
            "data": {"name": "Bob", "age": 30},
        }
        mock_get_row.return_value = mock_result

        result = await get_csv_row("test-session", "1")

        assert result["session_id"] == "test-session"
        assert result["row_index"] == 1
        mock_get_row.assert_called_once_with("test-session", 1)

    @pytest.mark.asyncio
    async def test_get_csv_row_invalid_index(self):
        """Test CSV row retrieval with invalid row index."""
        result = await get_csv_row("test-session", "not_a_number")

        assert "error" in result
        assert "Invalid row index" in result["error"]

    @pytest.mark.asyncio
    @patch("src.databeak.server._get_row_data")
    async def test_get_csv_row_tool_error(self, mock_get_row):
        """Test CSV row retrieval with ToolError."""
        mock_get_row.side_effect = ToolError("Row not found")

        result = await get_csv_row("test-session", "0")

        assert "error" in result
        assert "Row not found" in result["error"]

    @pytest.mark.asyncio
    @patch("src.databeak.server.get_session_manager")
    @patch("src.databeak.server.create_data_preview_with_indices")
    async def test_get_csv_preview_success(
        self, mock_preview, mock_get_manager, mock_session_with_data
    ):
        """Test successful CSV preview retrieval."""
        session, df = mock_session_with_data
        manager = MagicMock()
        manager.get_session.return_value = session
        mock_get_manager.return_value = manager
        mock_preview.return_value = {"preview_data": "test"}

        result = await get_csv_preview("test-session")

        assert result["session_id"] == "test-session"
        assert "coordinate_system" in result
        assert "preview_data" in result
        mock_preview.assert_called_once_with(df, 10)


class TestPromptFunctions:
    """Tests for prompt functions."""

    def test_analyze_csv_prompt_default(self):
        """Test CSV analysis prompt with default parameters."""
        result = analyze_csv_prompt("test-session")

        assert "test-session" in result
        assert "summary" in result
        assert "Data quality" in result
        assert "Statistical patterns" in result

    def test_analyze_csv_prompt_custom_type(self):
        """Test CSV analysis prompt with custom analysis type."""
        result = analyze_csv_prompt("test-session", "detailed")

        assert "test-session" in result
        assert "detailed" in result

    def test_data_cleaning_prompt(self):
        """Test data cleaning prompt."""
        result = data_cleaning_prompt("test-session")

        assert "test-session" in result
        assert "Missing values" in result
        assert "Duplicate rows" in result
        assert "Data type conversions" in result
        assert "Outliers" in result


class TestMainFunction:
    """Tests for main entry point function."""

    @patch("src.databeak.server.argparse.ArgumentParser")
    @patch("src.databeak.server.setup_structured_logging")
    @patch("src.databeak.server.set_correlation_id")
    @patch("src.databeak.server.logger")
    @patch("src.databeak.server.mcp")
    def test_main_stdio_transport(
        self, mock_mcp, mock_logger, mock_set_id, mock_setup_logging, mock_parser
    ):
        """Test main function with stdio transport."""
        # Mock argument parser
        mock_args = MagicMock()
        mock_args.transport = "stdio"
        mock_args.log_level = "INFO"
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        mock_set_id.return_value = "server-123"

        main()

        mock_setup_logging.assert_called_once_with("INFO")
        mock_set_id.assert_called_once()
        mock_logger.info.assert_called_once()
        mock_mcp.run.assert_called_once_with()

    @patch("src.databeak.server.argparse.ArgumentParser")
    @patch("src.databeak.server.setup_structured_logging")
    @patch("src.databeak.server.set_correlation_id")
    @patch("src.databeak.server.logger")
    @patch("src.databeak.server.mcp")
    def test_main_http_transport(
        self, mock_mcp, mock_logger, mock_set_id, mock_setup_logging, mock_parser
    ):
        """Test main function with HTTP transport."""
        mock_args = MagicMock()
        mock_args.transport = "http"
        mock_args.host = "localhost"
        mock_args.port = 8080
        mock_args.log_level = "DEBUG"
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        mock_set_id.return_value = "server-456"

        main()

        mock_setup_logging.assert_called_once_with("DEBUG")
        mock_mcp.run.assert_called_once_with(transport="http", host="localhost", port=8080)

    @patch("src.databeak.server.argparse.ArgumentParser")
    @patch("src.databeak.server.setup_structured_logging")
    @patch("src.databeak.server.set_correlation_id")
    @patch("src.databeak.server.logger")
    @patch("src.databeak.server.mcp")
    def test_main_sse_transport(
        self, mock_mcp, mock_logger, mock_set_id, mock_setup_logging, mock_parser
    ):
        """Test main function with SSE transport."""
        mock_args = MagicMock()
        mock_args.transport = "sse"
        mock_args.host = "0.0.0.0"
        mock_args.port = 9000
        mock_args.log_level = "WARNING"
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        main()

        mock_setup_logging.assert_called_once_with("WARNING")
        mock_mcp.run.assert_called_once_with(transport="sse", host="0.0.0.0", port=9000)


class TestServerModule:
    """Integration tests for the server module as a whole."""

    def test_server_imports(self):
        """Test that all necessary imports are available."""
        # Test that key functions are importable
        from src.databeak.server import _load_instructions, analyze_csv_prompt, get_csv_data, main

        assert callable(_load_instructions)
        assert callable(get_csv_data)
        assert callable(analyze_csv_prompt)
        assert callable(main)

    @patch("src.databeak.server.mcp")
    def test_server_initialization(self, mock_mcp):
        """Test that server initialization completes without errors."""
        # Import the module to trigger initialization

        # Verify that mount was called for each server
        expected_mounts = 7  # io, stats, discovery, validation, transformation, column, column_text
        assert mock_mcp.mount.call_count == expected_mounts
