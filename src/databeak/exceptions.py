"""Custom exceptions for DataBeak operations.

This module defines all custom exceptions for DataBeak with structured error details.
All exceptions use specific types to maintain type safety while allowing serialization.

Type Safety Notes:
- ErrorDetail values are constrained to serializable types used in practice
- InvalidParameterError.value uses Any because it receives arbitrary invalid input
- Base class details dict uses specific union type matching actual usage patterns
"""

from __future__ import annotations

from typing import Any

# Type alias for exception detail values - constrained to types actually used
ErrorDetail = str | int | list[str] | None


class DatabeakError(Exception):
    """Base exception with error details and serialization.

    All DataBeak exceptions inherit from this class and provide:
    - Structured error messages
    - Error codes for programmatic handling
    - Typed details dictionary for context

    The details dictionary accepts ErrorDetail values (str, int, list[str], or None)
    which covers all actual usage patterns in the codebase.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, ErrorDetail] | None = None,
    ):
        """Initialize with message, code, and details."""
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> dict[str, str | None | dict[str, ErrorDetail]]:
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
    """Invalid parameter value.

    Note on Type Safety:
    The `value` parameter uses `Any` type annotation because this exception is raised
    when validating user input, which can be of any type. The value is immediately
    converted to string for safe serialization, so the Any type is contained and
    justified for this validation error use case.
    """

    def __init__(
        self,
        parameter: str,
        value: Any,  # Any justified: receives arbitrary invalid input during validation
        expected: str | None = None,
    ):
        """Initialize with parameter details.

        Args:
            parameter: Name of the invalid parameter
            value: The invalid value (Any type as it comes from user input validation)
            expected: Optional description of expected value format

        """
        super().__init__(
            f"Invalid value for parameter '{parameter}': {value}",
            error_code="INVALID_PARAMETER",
            details={
                "parameter": parameter,
                "value": str(value),  # Converted to string for safe serialization
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
