"""Structured logging with correlation IDs and JSON formatting."""

from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Context variable for correlation ID tracking
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


class CorrelationFilter(logging.Filter):
    """Filter that adds correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record."""
        record.correlation_id = correlation_id.get() or "no-correlation"
        return True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured log output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "no-correlation"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id
        if hasattr(record, "operation_type"):
            log_data["operation_type"] = record.operation_type
        if hasattr(record, "user_context"):
            log_data["user_context"] = record.user_context

        return json.dumps(log_data, ensure_ascii=False)


# Implementation: Structured logging setup with JSON formatter and correlation filter
def setup_structured_logging(level: str = "INFO") -> None:
    """Set up structured logging with JSON formatting."""
    # Create structured formatter
    formatter = StructuredFormatter()

    # Create correlation filter
    correlation_filter = CorrelationFilter()

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(correlation_filter)

    root_logger.addHandler(console_handler)


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id.get() or "no-correlation"


# Implementation: Context variable correlation ID setting with UUID generation
def set_correlation_id(corr_id: str | None = None) -> str:
    """Set correlation ID for current context."""
    if corr_id is None:
        corr_id = str(uuid.uuid4())
    correlation_id.set(corr_id)
    return corr_id


def clear_correlation_id() -> None:
    """Clear the correlation ID from context."""
    correlation_id.set("")


class CorrelatedLogger:
    """Logger wrapper that includes correlation ID and structured fields."""

    def __init__(self, name: str):
        """Initialize correlated logger."""
        self.logger = logging.getLogger(name)

    def _log_with_context(self, level: int, msg: str, **kwargs: Any) -> None:
        """Log message with additional context."""
        extra = {"correlation_id": get_correlation_id(), **kwargs}
        self.logger.log(level, msg, extra=extra)

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message with context."""
        self._log_with_context(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log error message with context."""
        self._log_with_context(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, msg, **kwargs)


def get_logger(name: str) -> CorrelatedLogger:
    """Get a correlated logger instance."""
    return CorrelatedLogger(name)


def log_operation_start(operation: str, session_id: str | None = None, **context: Any) -> None:
    """Log the start of an operation."""
    logger = get_logger("databeak.operations")
    logger.info(
        f"Operation started: {operation}",
        session_id=session_id,
        operation_type=operation,
        operation_phase="start",
        **context,
    )


def log_operation_end(
    operation: str, session_id: str | None = None, success: bool = True, **context: Any
) -> None:
    """Log the end of an operation."""
    logger = get_logger("databeak.operations")
    level_method = logger.info if success else logger.error
    level_method(
        f"Operation {'completed' if success else 'failed'}: {operation}",
        session_id=session_id,
        operation_type=operation,
        operation_phase="end",
        operation_success=success,
        **context,
    )


def log_session_event(event: str, session_id: str, **context: Any) -> None:
    """Log a session-related event."""
    logger = get_logger("databeak.sessions")
    logger.info(f"Session event: {event}", session_id=session_id, event_type=event, **context)


# Convenience function for backward compatibility
def setup_logging(level: str = "INFO") -> None:
    """Set up logging (backward compatibility)."""
    setup_structured_logging(level)
