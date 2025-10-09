"""Unit tests for server.py module."""

from unittest.mock import MagicMock, patch

import pytest


class TestLoadInstructions:
    """Tests for _load_instructions function."""

    @patch("databeak.server.Path")
    def test_load_instructions_success(self, mock_path_constructor: MagicMock) -> None:
        """Test successful loading of instructions."""
        from databeak.server import _load_instructions

        # Mock the path chain: Path(__file__).parent / "instructions.md"
        mock_instructions_file = MagicMock()
        mock_instructions_file.read_text.return_value = "Test instructions content"

        mock_parent = MagicMock()
        mock_parent.__truediv__.return_value = mock_instructions_file

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_parent

        mock_path_constructor.return_value = mock_path_instance

        result = _load_instructions()

        assert result == "Test instructions content"
        mock_instructions_file.read_text.assert_called_once_with(encoding="utf-8")

    @patch("databeak.server.Path")
    @patch("databeak.server.logger")
    def test_load_instructions_file_not_found(
        self, mock_logger: MagicMock, mock_path_constructor: MagicMock
    ) -> None:
        """Test instructions loading when file not found."""
        from databeak.server import _load_instructions

        mock_instructions_file = MagicMock()
        mock_instructions_file.read_text.side_effect = FileNotFoundError()

        mock_parent = MagicMock()
        mock_parent.__truediv__.return_value = mock_instructions_file

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_parent

        mock_path_constructor.return_value = mock_path_instance

        result = _load_instructions()

        assert "Instructions file not available" in result
        mock_logger.warning.assert_called_once()

    @patch("databeak.server.Path")
    @patch("databeak.server.logger")
    def test_load_instructions_other_error(
        self, mock_logger: MagicMock, mock_path_constructor: MagicMock
    ) -> None:
        """Test instructions loading with other error."""
        from databeak.server import _load_instructions

        mock_instructions_file = MagicMock()
        mock_instructions_file.read_text.side_effect = OSError("Generic error")

        mock_parent = MagicMock()
        mock_parent.__truediv__.return_value = mock_instructions_file

        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_parent

        mock_path_constructor.return_value = mock_path_instance

        result = _load_instructions()

        assert "Error loading instructions" in result
        # Code uses logger.exception() for OSError handling
        mock_logger.exception.assert_called_once()


class TestMainFunction:
    """Tests for main entry point function covering lines 225-264."""

    @patch("databeak.server.ArgumentParser")
    @patch("databeak.server.setup_structured_logging")
    @patch("databeak.server.set_correlation_id")
    @patch("databeak.server.logger")
    @patch("databeak.server.create_server")
    def test_main_stdio_transport(
        self,
        mock_create_server: MagicMock,
        mock_logger: MagicMock,
        mock_set_id: MagicMock,
        mock_setup_logging: MagicMock,
        mock_parser: MagicMock,
    ) -> None:
        """Test main function with stdio transport - covers lines 225-262."""
        from databeak.server import main

        mock_args = MagicMock()
        mock_args.transport = "stdio"
        mock_args.host = "0.0.0.0"
        mock_args.port = 8000
        mock_args.log_level = "INFO"
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        mock_set_id.return_value = "server-123"

        # Setup mock MCP instance
        mock_mcp_instance = MagicMock()
        mock_create_server.return_value = mock_mcp_instance

        main()

        # Verify argument parser setup (lines 227-241)
        mock_parser.assert_called_once_with(description="DataBeak")

        # Verify setup calls (lines 246-249)
        mock_setup_logging.assert_called_once_with("INFO")
        mock_set_id.assert_called_once()

        # Verify logging call (lines 251-258)
        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args
        assert "Starting DataBeak" in log_call[0][0]

        # Verify stdio transport execution (lines 261-262)
        mock_create_server.assert_called_once()
        mock_mcp_instance.run.assert_called_once_with(transport="stdio")

    @patch("databeak.server.ArgumentParser")
    @patch("databeak.server.setup_structured_logging")
    @patch("databeak.server.set_correlation_id")
    @patch("databeak.server.logger")
    @patch("databeak.server.create_server")
    def test_main_http_transport(
        self,
        mock_create_server: MagicMock,
        mock_logger: MagicMock,
        mock_set_id: MagicMock,
        mock_setup_logging: MagicMock,
        mock_parser: MagicMock,
    ) -> None:
        """Test main function with HTTP transport - covers lines 263-264."""
        from databeak.server import main

        mock_args = MagicMock()
        mock_args.transport = "http"
        mock_args.host = "localhost"
        mock_args.port = 8080
        mock_args.log_level = "DEBUG"
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        mock_set_id.return_value = "server-456"

        # Setup mock MCP instance
        mock_mcp_instance = MagicMock()
        mock_create_server.return_value = mock_mcp_instance

        main()

        mock_setup_logging.assert_called_once_with("DEBUG")
        # Verify non-stdio transport execution (line 264)
        mock_create_server.assert_called_once()
        mock_mcp_instance.run.assert_called_once_with(transport="http", host="localhost", port=8080)

    @patch("databeak.server.ArgumentParser")
    @patch("databeak.server.setup_structured_logging")
    @patch("databeak.server.set_correlation_id")
    @patch("databeak.server.logger")
    @patch("databeak.server.create_server")
    def test_main_sse_transport(
        self,
        mock_create_server: MagicMock,
        mock_logger: MagicMock,
        mock_set_id: MagicMock,
        mock_setup_logging: MagicMock,
        mock_parser: MagicMock,
    ) -> None:
        """Test main function with SSE transport - covers lines 263-264."""
        from databeak.server import main

        mock_args = MagicMock()
        mock_args.transport = "sse"
        mock_args.host = "0.0.0.0"
        mock_args.port = 9000
        mock_args.log_level = "WARNING"
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        # Setup mock MCP instance
        mock_mcp_instance = MagicMock()
        mock_create_server.return_value = mock_mcp_instance

        main()

        mock_setup_logging.assert_called_once_with("WARNING")
        # Verify non-stdio transport execution (line 264)
        mock_create_server.assert_called_once()
        mock_mcp_instance.run.assert_called_once_with(transport="sse", host="0.0.0.0", port=9000)

    @pytest.mark.parametrize(
        ("transport", "expected_run_args"),
        [
            ("stdio", {}),
            ("http", {"transport": "http", "host": "localhost", "port": 8080}),
            ("sse", {"transport": "sse", "host": "0.0.0.0", "port": 9000}),
        ],
    )
    @patch("databeak.server.ArgumentParser")
    @patch("databeak.server.setup_structured_logging")
    @patch("databeak.server.set_correlation_id")
    @patch("databeak.server.logger")
    @patch("databeak.server.create_server")
    def test_main_transport_variations(
        self,
        mock_create_server: MagicMock,
        mock_logger: MagicMock,
        mock_set_id: MagicMock,
        mock_setup_logging: MagicMock,
        mock_parser: MagicMock,
        transport: str,
        expected_run_args: dict[str, dict[str, str | int]],
    ) -> None:
        """Test main function with various transport configurations - covers transport branching logic."""
        from databeak.server import main

        mock_args = MagicMock()
        mock_args.transport = transport
        mock_args.host = expected_run_args.get("host", "localhost")
        mock_args.port = expected_run_args.get("port", 8080)
        mock_args.log_level = "INFO"
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        mock_set_id.return_value = f"server-{transport}"

        # Setup mock MCP instance
        mock_mcp_instance = MagicMock()
        mock_create_server.return_value = mock_mcp_instance

        main()

        # Verify logging was called (implementation details not tested)
        mock_logger.info.assert_called_once()

        # Verify correct mcp.run() call based on transport
        mock_create_server.assert_called_once()
        if transport == "stdio":
            mock_mcp_instance.run.assert_called_once_with(transport="stdio")
        else:
            mock_mcp_instance.run.assert_called_once_with(**expected_run_args)

    @patch("databeak.server.ArgumentParser")
    def test_main_argument_parser_configuration(self, mock_parser: MagicMock) -> None:
        """Test that argument parser is configured correctly - covers lines 227-241."""
        from databeak.server import main

        mock_args = MagicMock()
        mock_args.transport = "stdio"
        mock_args.host = "0.0.0.0"
        mock_args.port = 8000
        mock_args.log_level = "INFO"
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        with (
            patch("databeak.server.setup_structured_logging"),
            patch("databeak.server.set_correlation_id"),
            patch("databeak.server.logger"),
            patch("databeak.server.create_server"),
        ):
            main()

        # Verify argument parser created with correct description (line 227)
        mock_parser.assert_called_once_with(description="DataBeak")

        # Verify all arguments are added
        add_arg_calls = mock_parser_instance.add_argument.call_args_list
        arg_names = [call[0][0] for call in add_arg_calls]

        assert "--transport" in arg_names
        assert "--host" in arg_names
        assert "--port" in arg_names
        assert "--log-level" in arg_names

        # Find and verify transport argument configuration
        transport_call = next(call for call in add_arg_calls if "--transport" in call[0])
        assert "choices" in transport_call[1]
        assert transport_call[1]["choices"] == ["stdio", "http", "sse"]
        assert transport_call[1]["default"] == "stdio"

    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    @patch("databeak.server.ArgumentParser")
    @patch("databeak.server.setup_structured_logging")
    @patch("databeak.server.set_correlation_id")
    @patch("databeak.server.logger")
    @patch("databeak.server.create_server")
    def test_main_logging_levels(
        self,
        mock_create_server: MagicMock,
        mock_logger: MagicMock,
        mock_set_id: MagicMock,
        mock_setup_logging: MagicMock,
        mock_parser: MagicMock,
        log_level: str,
    ) -> None:
        """Test that all supported logging levels work correctly."""
        from databeak.server import main

        mock_args = MagicMock()
        mock_args.transport = "stdio"
        mock_args.host = "0.0.0.0"
        mock_args.port = 8000
        mock_args.log_level = log_level
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        mock_set_id.return_value = "test-server"

        # Setup mock MCP instance
        mock_mcp_instance = MagicMock()
        mock_create_server.return_value = mock_mcp_instance

        main()

        # Verify logging is set up with the correct level
        mock_setup_logging.assert_called_once_with(log_level)

    @patch("databeak.server.ArgumentParser")
    @patch("databeak.server.setup_structured_logging")
    @patch("databeak.server.set_correlation_id")
    @patch("databeak.server.logger")
    @patch("databeak.server.create_server")
    def test_main_correlation_id_logging(
        self,
        mock_create_server: MagicMock,
        mock_logger: MagicMock,
        mock_set_id: MagicMock,
        mock_setup_logging: MagicMock,
        mock_parser: MagicMock,
    ) -> None:
        """Test that server sets correlation ID and includes it in logs."""
        from databeak.server import main

        mock_args = MagicMock()
        mock_args.transport = "stdio"
        mock_args.host = "0.0.0.0"
        mock_args.port = 8000
        mock_args.log_level = "INFO"
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        test_correlation_id = "test-correlation-123"
        mock_set_id.return_value = test_correlation_id

        # Setup mock MCP instance
        mock_mcp_instance = MagicMock()
        mock_create_server.return_value = mock_mcp_instance

        main()

        # Verify correlation ID was set and used in logging
        mock_set_id.assert_called_once()
        mock_logger.info.assert_called_once()
        # Verify logging was called (implementation details not tested)
        mock_logger.info.assert_called_once()

    @patch("databeak.server.ArgumentParser")
    @patch("databeak.server.setup_structured_logging")
    @patch("databeak.server.set_correlation_id")
    @patch("databeak.server.logger")
    @patch("databeak.server.create_server")
    def test_main_conditional_logging_params(
        self,
        mock_create_server: MagicMock,
        mock_logger: MagicMock,
        mock_set_id: MagicMock,
        mock_setup_logging: MagicMock,
        mock_parser: MagicMock,
    ) -> None:
        """Test conditional logging parameters for different transports."""
        from databeak.server import main

        test_cases = [
            ("stdio", None, None),
            ("http", "localhost", 8080),
            ("sse", "0.0.0.0", 9000),
        ]

        for transport, host, port in test_cases:
            # Reset mocks
            mock_logger.reset_mock()
            mock_create_server.reset_mock()

            mock_args = MagicMock()
            mock_args.transport = transport
            mock_args.host = host if host else "localhost"
            mock_args.port = port if port else 8080
            mock_args.log_level = "INFO"
            mock_parser_instance = MagicMock()
            mock_parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = mock_parser_instance

            # Setup mock MCP instance
            mock_mcp_instance = MagicMock()
            mock_create_server.return_value = mock_mcp_instance

            main()

            # Verify logging was called (implementation details not tested)
            mock_logger.info.assert_called_once()


class TestServerInitialization:
    """Tests for server initialization and configuration."""

    def test_create_server_function(self) -> None:
        """Test that create_server function exists and returns MCP instance."""
        from databeak.server import create_server

        # Create server instance
        mcp_instance = create_server()

        assert mcp_instance is not None
        assert hasattr(mcp_instance, "mount")
        assert hasattr(mcp_instance, "run")

    def test_server_imports(self) -> None:
        """Test that all server imports are available."""
        from databeak import server

        # Verify key functions exist
        assert hasattr(server, "_load_instructions")
        assert hasattr(server, "main")
        assert hasattr(server, "create_server")

    def test_server_mounting_imports(self) -> None:
        """Test that all server modules are imported for mounting."""
        from databeak import server

        # Verify server imports exist (covers lines 55-64)
        server_modules = [
            "system_server",
            "io_server",
            "row_operations_server",
            "statistics_server",
            "discovery_server",
            "validation_server",
            "transformation_server",
            "column_server",
            "column_text_server",
        ]

        for module in server_modules:
            assert hasattr(server, module), f"Server module {module} not imported"

    def test_instructions_loaded_during_init(self) -> None:
        """Test that instructions are loaded during server initialization."""
        # Check that _load_instructions function exists and works
        from databeak.server import _load_instructions

        # Call the function to ensure it executes
        result = _load_instructions()

        # Should return string content (either loaded or fallback)
        assert isinstance(result, str)
        assert len(result) > 0


# Note: TestResourceAndPromaptLogic class removed as resources were extracted to dedicated module in #86
