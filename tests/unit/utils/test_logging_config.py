"""Test coverage for logging configuration module."""

from __future__ import annotations

import json
import logging
import uuid
from unittest.mock import MagicMock, patch

from databeak.utils.logging_config import (
    CorrelationFilter,
    StructuredFormatter,
    clear_correlation_id,
    correlation_id,
    get_correlation_id,
    set_correlation_id,
    setup_logging,
    setup_structured_logging,
)


class TestCorrelationFilter:
    """Test correlation filter functionality."""

    def test_filter_adds_correlation_id_when_set(self) -> None:
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

    def test_filter_adds_default_correlation_id_when_none_set(self) -> None:
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

    def test_filter_adds_default_when_empty_correlation_id(self) -> None:
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

    def test_format_basic_log_record(self) -> None:
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

    def test_format_with_exception_info(self) -> None:
        """Test formatting with exception information."""
        formatter = StructuredFormatter()

        try:
            msg = "Test exception"
            raise ValueError(msg)
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

    def test_format_with_session_id(self) -> None:
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

    def test_format_with_operation_type(self) -> None:
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

    def test_format_with_user_context(self) -> None:
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

    def test_format_without_correlation_id_attribute(self) -> None:
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

    def test_format_without_extra_attributes(self) -> None:
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

    def test_format_without_exception_info(self) -> None:
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

    def test_setup_structured_logging_default_level(self) -> None:
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

    def test_setup_structured_logging_custom_level(self) -> None:
        """Test setup with custom log level."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            # Execute
            setup_structured_logging("DEBUG")

            # Verify
            mock_logger.setLevel.assert_called_once_with(logging.DEBUG)

    def test_setup_structured_logging_clears_existing_handlers(self) -> None:
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

    def test_setup_structured_logging_adds_console_handler(self) -> None:
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

    def test_get_correlation_id_when_set(self) -> None:
        """Test getting correlation ID when one is set."""
        test_id = "test-correlation-456"
        set_correlation_id(test_id)

        result = get_correlation_id()

        assert result == test_id
        clear_correlation_id()

    def test_get_correlation_id_when_not_set(self) -> None:
        """Test getting correlation ID when none is set."""
        clear_correlation_id()

        result = get_correlation_id()

        assert result == "no-correlation"

    def test_get_correlation_id_when_empty_string(self) -> None:
        """Test getting correlation ID when set to empty string."""
        correlation_id.set("")

        result = get_correlation_id()

        assert result == "no-correlation"

    def test_set_correlation_id_with_provided_id(self) -> None:
        """Test setting correlation ID with provided value."""
        test_id = "custom-correlation-789"

        result = set_correlation_id(test_id)

        assert result == test_id
        assert get_correlation_id() == test_id
        clear_correlation_id()

    def test_set_correlation_id_with_none_generates_uuid(self) -> None:
        """Test setting correlation ID with None generates UUID."""
        result = set_correlation_id(None)

        # Verify it's a valid UUID format
        uuid.UUID(result)  # Will raise if not valid
        assert get_correlation_id() == result
        clear_correlation_id()

    def test_set_correlation_id_no_argument_generates_uuid(self) -> None:
        """Test setting correlation ID with no argument generates UUID."""
        result = set_correlation_id()

        # Verify it's a valid UUID format
        uuid.UUID(result)  # Will raise if not valid
        assert get_correlation_id() == result
        clear_correlation_id()

    def test_clear_correlation_id(self) -> None:
        """Test clearing correlation ID."""
        set_correlation_id("test-id")

        clear_correlation_id()

        assert get_correlation_id() == "no-correlation"


# NOTE: TestGetLogger, TestOperationLogging, and TestSessionLogging removed
# since we now use standard Python logging.Logger which is well-tested by Python itself.


class TestBackwardCompatibility:
    """Test backward compatibility functions."""

    def test_setup_logging_calls_setup_structured_logging(self) -> None:
        """Test setup_logging calls setup_structured_logging for backward compatibility."""
        with patch("databeak.utils.logging_config.setup_structured_logging") as mock_setup:
            setup_logging("WARNING")

            mock_setup.assert_called_once_with("WARNING")

    def test_setup_logging_default_level(self) -> None:
        """Test setup_logging with default level."""
        with patch("databeak.utils.logging_config.setup_structured_logging") as mock_setup:
            setup_logging()

            mock_setup.assert_called_once_with("INFO")
