"""Test coverage for logging configuration module."""

from __future__ import annotations

import json
import logging
import uuid
from unittest.mock import MagicMock, patch

from src.databeak.utils.logging_config import (
    CorrelatedLogger,
    CorrelationFilter,
    StructuredFormatter,
    clear_correlation_id,
    correlation_id,
    get_correlation_id,
    get_logger,
    log_operation_end,
    log_operation_start,
    log_session_event,
    set_correlation_id,
    setup_logging,
    setup_structured_logging,
)


class TestCorrelationFilter:
    """Test correlation filter functionality."""

    def test_filter_adds_correlation_id_when_set(self):
        """Test filter adds correlation ID when one is set."""
        # Setup
        test_id = "test-correlation-123"
        set_correlation_id(test_id)

        filter_instance = CorrelationFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # Execute
        result = filter_instance.filter(record)

        # Verify
        assert result is True
        assert record.correlation_id == test_id  # type: ignore[attr-defined]

        # Cleanup
        clear_correlation_id()

    def test_filter_adds_default_correlation_id_when_none_set(self):
        """Test filter adds default when no correlation ID is set."""
        # Setup
        clear_correlation_id()

        filter_instance = CorrelationFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # Execute
        result = filter_instance.filter(record)

        # Verify
        assert result is True
        assert record.correlation_id == "no-correlation"  # type: ignore[attr-defined]

    def test_filter_adds_default_when_empty_correlation_id(self):
        """Test filter adds default when correlation ID is empty string."""
        # Setup - set empty string (falsy value)
        correlation_id.set("")

        filter_instance = CorrelationFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # Execute
        result = filter_instance.filter(record)

        # Verify
        assert result is True
        assert record.correlation_id == "no-correlation"  # type: ignore[attr-defined]


class TestStructuredFormatter:
    """Test structured JSON formatter."""

    def test_format_basic_log_record(self):
        """Test basic log record formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.correlation_id = "test-123"

        # Execute
        result = formatter.format(record)

        # Verify
        log_data = json.loads(result)
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert log_data["correlation_id"] == "test-123"
        assert log_data["module"] == "path"
        assert log_data["function"] == "test_function"
        assert log_data["line"] == 42
        assert "timestamp" in log_data

    def test_format_with_exception_info(self):
        """Test formatting with exception information."""
        formatter = StructuredFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="/test/path.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
                func="test_function",
            )
            record.correlation_id = "test-123"

            # Execute
            result = formatter.format(record)

            # Verify
            log_data = json.loads(result)
            assert "exception" in log_data
            assert "ValueError: Test exception" in log_data["exception"]

    def test_format_with_session_id(self):
        """Test formatting with session ID."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Session event",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.correlation_id = "test-123"
        record.session_id = "session-456"

        # Execute
        result = formatter.format(record)

        # Verify
        log_data = json.loads(result)
        assert log_data["session_id"] == "session-456"

    def test_format_with_operation_type(self):
        """Test formatting with operation type."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Operation event",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.correlation_id = "test-123"
        record.operation_type = "data_load"

        # Execute
        result = formatter.format(record)

        # Verify
        log_data = json.loads(result)
        assert log_data["operation_type"] == "data_load"

    def test_format_with_user_context(self):
        """Test formatting with user context."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="User event",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.correlation_id = "test-123"
        record.user_context = {"user_id": "user-789"}

        # Execute
        result = formatter.format(record)

        # Verify
        log_data = json.loads(result)
        assert log_data["user_context"] == {"user_id": "user-789"}

    def test_format_without_correlation_id_attribute(self):
        """Test formatting when correlation_id attribute is missing."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )
        # Don't set correlation_id attribute

        # Execute
        result = formatter.format(record)

        # Verify
        log_data = json.loads(result)
        assert log_data["correlation_id"] == "no-correlation"

    def test_format_without_extra_attributes(self):
        """Test formatting when no extra attributes are present."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.correlation_id = "test-123"
        # Don't set session_id, operation_type, or user_context

        # Execute
        result = formatter.format(record)

        # Verify
        log_data = json.loads(result)
        assert "session_id" not in log_data
        assert "operation_type" not in log_data
        assert "user_context" not in log_data
        assert log_data["correlation_id"] == "test-123"

    def test_format_without_exception_info(self):
        """Test formatting when no exception info is present."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )
        record.correlation_id = "test-123"

        # Execute
        result = formatter.format(record)

        # Verify
        log_data = json.loads(result)
        assert "exception" not in log_data
        assert log_data["correlation_id"] == "test-123"


class TestSetupStructuredLogging:
    """Test structured logging setup."""

    def test_setup_structured_logging_default_level(self):
        """Test setup with default INFO level."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.handlers = []  # No existing handlers
            mock_get_logger.return_value = mock_logger

            # Execute
            setup_structured_logging()

            # Verify
            mock_logger.setLevel.assert_called_once_with(logging.INFO)
            assert mock_logger.addHandler.called

    def test_setup_structured_logging_custom_level(self):
        """Test setup with custom log level."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            # Execute
            setup_structured_logging("DEBUG")

            # Verify
            mock_logger.setLevel.assert_called_once_with(logging.DEBUG)

    def test_setup_structured_logging_clears_existing_handlers(self):
        """Test that existing handlers are cleared."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            existing_handler = MagicMock()
            mock_logger.handlers = [existing_handler]
            mock_get_logger.return_value = mock_logger

            # Execute
            setup_structured_logging()

            # Verify
            mock_logger.removeHandler.assert_called_once_with(existing_handler)

    def test_setup_structured_logging_adds_console_handler(self):
        """Test that console handler is added with proper configuration."""
        with (
            patch("logging.getLogger") as mock_get_logger,
            patch("logging.StreamHandler") as mock_stream_handler,
        ):
            mock_logger = MagicMock()
            mock_logger.handlers = []
            mock_get_logger.return_value = mock_logger
            mock_handler = MagicMock()
            mock_stream_handler.return_value = mock_handler

            # Execute
            setup_structured_logging()

            # Verify
            mock_stream_handler.assert_called_once()
            mock_handler.setFormatter.assert_called_once()
            mock_handler.addFilter.assert_called_once()
            mock_logger.addHandler.assert_called_once_with(mock_handler)


class TestCorrelationIdManagement:
    """Test correlation ID context management."""

    def test_get_correlation_id_when_set(self):
        """Test getting correlation ID when one is set."""
        test_id = "test-correlation-456"
        set_correlation_id(test_id)

        result = get_correlation_id()

        assert result == test_id
        clear_correlation_id()

    def test_get_correlation_id_when_not_set(self):
        """Test getting correlation ID when none is set."""
        clear_correlation_id()

        result = get_correlation_id()

        assert result == "no-correlation"

    def test_get_correlation_id_when_empty_string(self):
        """Test getting correlation ID when set to empty string."""
        correlation_id.set("")

        result = get_correlation_id()

        assert result == "no-correlation"

    def test_set_correlation_id_with_provided_id(self):
        """Test setting correlation ID with provided value."""
        test_id = "custom-correlation-789"

        result = set_correlation_id(test_id)

        assert result == test_id
        assert get_correlation_id() == test_id
        clear_correlation_id()

    def test_set_correlation_id_with_none_generates_uuid(self):
        """Test setting correlation ID with None generates UUID."""
        result = set_correlation_id(None)

        # Verify it's a valid UUID format
        uuid.UUID(result)  # Will raise if not valid
        assert get_correlation_id() == result
        clear_correlation_id()

    def test_set_correlation_id_no_argument_generates_uuid(self):
        """Test setting correlation ID with no argument generates UUID."""
        result = set_correlation_id()

        # Verify it's a valid UUID format
        uuid.UUID(result)  # Will raise if not valid
        assert get_correlation_id() == result
        clear_correlation_id()

    def test_clear_correlation_id(self):
        """Test clearing correlation ID."""
        set_correlation_id("test-id")

        clear_correlation_id()

        assert get_correlation_id() == "no-correlation"


class TestCorrelatedLogger:
    """Test correlated logger wrapper."""

    def test_correlated_logger_initialization(self):
        """Test correlated logger initialization."""
        logger = CorrelatedLogger("test.logger")

        assert logger.logger.name == "test.logger"

    def test_debug_logging(self):
        """Test debug level logging."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance

            logger = CorrelatedLogger("test.logger")
            set_correlation_id("test-123")

            logger.debug("Debug message", session_id="session-456")

            mock_logger_instance.log.assert_called_once_with(
                logging.DEBUG,
                "Debug message",
                extra={"correlation_id": "test-123", "session_id": "session-456"},
            )
            clear_correlation_id()

    def test_info_logging(self):
        """Test info level logging."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance

            logger = CorrelatedLogger("test.logger")
            set_correlation_id("test-123")

            logger.info("Info message", operation_type="test_op")

            mock_logger_instance.log.assert_called_once_with(
                logging.INFO,
                "Info message",
                extra={"correlation_id": "test-123", "operation_type": "test_op"},
            )
            clear_correlation_id()

    def test_warning_logging(self):
        """Test warning level logging."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance

            logger = CorrelatedLogger("test.logger")
            set_correlation_id("test-123")

            logger.warning("Warning message")

            mock_logger_instance.log.assert_called_once_with(
                logging.WARNING, "Warning message", extra={"correlation_id": "test-123"}
            )
            clear_correlation_id()

    def test_error_logging(self):
        """Test error level logging."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance

            logger = CorrelatedLogger("test.logger")
            set_correlation_id("test-123")

            logger.error("Error message", error_code="E001")

            mock_logger_instance.log.assert_called_once_with(
                logging.ERROR,
                "Error message",
                extra={"correlation_id": "test-123", "error_code": "E001"},
            )
            clear_correlation_id()

    def test_critical_logging(self):
        """Test critical level logging."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance

            logger = CorrelatedLogger("test.logger")
            set_correlation_id("test-123")

            logger.critical("Critical message")

            mock_logger_instance.log.assert_called_once_with(
                logging.CRITICAL, "Critical message", extra={"correlation_id": "test-123"}
            )
            clear_correlation_id()


class TestGetLogger:
    """Test logger factory function."""

    def test_get_logger_returns_correlated_logger(self):
        """Test get_logger returns CorrelatedLogger instance."""
        logger = get_logger("test.module")

        assert isinstance(logger, CorrelatedLogger)
        assert logger.logger.name == "test.module"


class TestOperationLogging:
    """Test operation logging convenience functions."""

    def test_log_operation_start_with_session_id(self):
        """Test logging operation start with session ID."""
        with patch("src.databeak.utils.logging_config.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            log_operation_start("data_load", session_id="session-123", user_id="user-456")

            mock_get_logger.assert_called_once_with("databeak.operations")
            mock_logger.info.assert_called_once_with(
                "Operation started: data_load",
                session_id="session-123",
                operation_type="data_load",
                operation_phase="start",
                user_id="user-456",
            )

    def test_log_operation_start_without_session_id(self):
        """Test logging operation start without session ID."""
        with patch("src.databeak.utils.logging_config.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            log_operation_start("data_transform")

            mock_logger.info.assert_called_once_with(
                "Operation started: data_transform",
                session_id=None,
                operation_type="data_transform",
                operation_phase="start",
            )

    def test_log_operation_end_success(self):
        """Test logging successful operation end."""
        with patch("src.databeak.utils.logging_config.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            log_operation_end("data_export", session_id="session-789", success=True, records=100)

            mock_get_logger.assert_called_once_with("databeak.operations")
            mock_logger.info.assert_called_once_with(
                "Operation completed: data_export",
                session_id="session-789",
                operation_type="data_export",
                operation_phase="end",
                operation_success=True,
                records=100,
            )

    def test_log_operation_end_failure(self):
        """Test logging failed operation end."""
        with patch("src.databeak.utils.logging_config.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            log_operation_end("data_validation", success=False, error="Invalid format")

            mock_logger.error.assert_called_once_with(
                "Operation failed: data_validation",
                session_id=None,
                operation_type="data_validation",
                operation_phase="end",
                operation_success=False,
                error="Invalid format",
            )

    def test_log_operation_end_default_success(self):
        """Test logging operation end with default success=True."""
        with patch("src.databeak.utils.logging_config.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            log_operation_end("cleanup_task")

            mock_logger.info.assert_called_once_with(
                "Operation completed: cleanup_task",
                session_id=None,
                operation_type="cleanup_task",
                operation_phase="end",
                operation_success=True,
            )


class TestSessionLogging:
    """Test session logging convenience function."""

    def test_log_session_event(self):
        """Test logging session events."""
        with patch("src.databeak.utils.logging_config.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            log_session_event("created", "session-abc", user_id="user-123", source="api")

            mock_get_logger.assert_called_once_with("databeak.sessions")
            mock_logger.info.assert_called_once_with(
                "Session event: created",
                session_id="session-abc",
                event_type="created",
                user_id="user-123",
                source="api",
            )


class TestBackwardCompatibility:
    """Test backward compatibility functions."""

    def test_setup_logging_calls_setup_structured_logging(self):
        """Test setup_logging calls setup_structured_logging for backward compatibility."""
        with patch("src.databeak.utils.logging_config.setup_structured_logging") as mock_setup:
            setup_logging("WARNING")

            mock_setup.assert_called_once_with("WARNING")

    def test_setup_logging_default_level(self):
        """Test setup_logging with default level."""
        with patch("src.databeak.utils.logging_config.setup_structured_logging") as mock_setup:
            setup_logging()

            mock_setup.assert_called_once_with("INFO")
