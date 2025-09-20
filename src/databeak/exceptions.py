"""Custom exceptions for DataBeak operations."""

from __future__ import annotations

from typing import Any


class DatabeakError(Exception):
    """Base exception with error details and serialization."""

    def __init__(self, message: str, error_code: str | None = None, details: dict | None = None):
        """Initialize with message, code, and details."""
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class SessionError(DatabeakError):
    """Session management errors."""


class SessionNotFoundError(SessionError):
    """Session not found or expired."""

    def __init__(self, session_id: str):
        """Initialize with session ID."""
        super().__init__(
            f"Session '{session_id}' not found or expired",
            error_code="SESSION_NOT_FOUND",
            details={"session_id": session_id},
        )


class SessionExpiredError(SessionError):
    """Session has expired."""

    def __init__(self, session_id: str):
        """Initialize with session ID."""
        super().__init__(
            f"Session '{session_id}' has expired",
            error_code="SESSION_EXPIRED",
            details={"session_id": session_id},
        )


class DataError(DatabeakError):
    """Data validation and processing errors."""


class NoDataLoadedError(DataError):
    """No data loaded in session."""

    def __init__(self, session_id: str):
        """Initialize with session ID."""
        super().__init__(
            f"No data loaded in session '{session_id}'",
            error_code="NO_DATA_LOADED",
            details={"session_id": session_id},
        )


class ColumnNotFoundError(DataError):
    """Column not found in dataset."""

    def __init__(self, column_name: str, available_columns: list[str] | None = None):
        """Initialize with column name and available columns."""
        super().__init__(
            f"Column '{column_name}' not found",
            error_code="COLUMN_NOT_FOUND",
            details={
                "column_name": column_name,
                "available_columns": available_columns or [],
            },
        )


class InvalidRowIndexError(DataError):
    """Row index is invalid or out of bounds."""

    def __init__(self, row_index: int, max_index: int | None = None):
        """Initialize with row index and bounds."""
        super().__init__(
            f"Row index {row_index} is invalid",
            error_code="INVALID_ROW_INDEX",
            details={"row_index": row_index, "max_index": max_index},
        )


class DataValidationError(DataError):
    """Data validation failed."""

    def __init__(self, message: str, validation_errors: list | None = None):
        """Initialize with validation details."""
        super().__init__(
            message,
            error_code="DATA_VALIDATION_ERROR",
            details={"validation_errors": validation_errors or []},
        )


class FileError(DatabeakError):
    """File access and format errors."""


class DataBeakFileNotFoundError(FileError):
    """File does not exist."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        super().__init__(
            f"File not found: {file_path}",
            error_code="FILE_NOT_FOUND",
            details={"file_path": file_path},
        )


class FilePermissionError(FileError):
    """File permission denied."""

    def __init__(self, file_path: str, operation: str):
        """Initialize with file path and operation."""
        super().__init__(
            f"Permission denied for {operation} operation on file: {file_path}",
            error_code="FILE_PERMISSION_ERROR",
            details={"file_path": file_path, "operation": operation},
        )


class FileFormatError(FileError):
    """File format is invalid or unsupported."""

    def __init__(self, file_path: str, expected_format: str | None = None):
        """Initialize with file path and expected format."""
        super().__init__(
            f"Invalid file format: {file_path}",
            error_code="FILE_FORMAT_ERROR",
            details={"file_path": file_path, "expected_format": expected_format},
        )


class OperationError(DatabeakError):
    """Data operation execution errors."""


class InvalidOperationError(OperationError):
    """Operation cannot be performed in current state."""

    def __init__(self, operation: str, reason: str):
        """Initialize with operation and reason."""
        super().__init__(
            f"Cannot perform operation '{operation}': {reason}",
            error_code="INVALID_OPERATION",
            details={"operation": operation, "reason": reason},
        )


class ParameterError(DatabeakError):
    """Function parameter validation errors."""


class InvalidParameterError(ParameterError):
    """Invalid parameter value."""

    def __init__(  # type: ignore[explicit-any]
        self,
        parameter: str,
        value: Any,
        expected: str | None = None,
    ):  # Any justified: can receive any invalid value type
        """Initialize with parameter details."""
        super().__init__(
            f"Invalid value for parameter '{parameter}': {value}",
            error_code="INVALID_PARAMETER",
            details={
                "parameter": parameter,
                "value": str(value),
                "expected": expected,
            },
        )


class MissingParameterError(ParameterError):
    """Required parameter is missing."""

    def __init__(self, parameter: str):
        """Initialize with parameter name."""
        super().__init__(
            f"Required parameter '{parameter}' is missing",
            error_code="MISSING_PARAMETER",
            details={"parameter": parameter},
        )


class HistoryError(DatabeakError):
    """Operation history management errors."""


class HistoryNotEnabledError(HistoryError):
    """History tracking is not enabled."""

    def __init__(self, session_id: str):
        """Initialize with session ID."""
        super().__init__(
            f"History is not enabled for session '{session_id}'",
            error_code="HISTORY_NOT_ENABLED",
            details={"session_id": session_id},
        )


class InvalidHistoryOperationError(HistoryError):
    """Invalid history operation."""

    def __init__(self, operation: str, reason: str):
        """Initialize with operation and reason."""
        super().__init__(
            f"Invalid history operation '{operation}': {reason}",
            error_code="INVALID_HISTORY_OPERATION",
            details={"operation": operation, "reason": reason},
        )


class AutoSaveError(DatabeakError):
    """Auto-save configuration and execution errors."""


class AutoSaveConfigError(AutoSaveError):
    """Auto-save configuration error."""

    def __init__(self, message: str, config_details: dict | None = None):
        """Initialize with configuration details."""
        super().__init__(
            f"Auto-save configuration error: {message}",
            error_code="AUTO_SAVE_CONFIG_ERROR",
            details=config_details or {},
        )
