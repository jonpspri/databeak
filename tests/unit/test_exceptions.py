"""Unit tests for exception classes."""

import pytest

from databeak.exceptions import (
    ColumnNotFoundError,
    DatabeakError,
    DataBeakFileNotFoundError,
    FileFormatError,
    FilePermissionError,
    InvalidOperationError,
    InvalidParameterError,
    InvalidRowIndexError,
    MissingParameterError,
    OperationError,
    SessionExpiredError,
    SessionNotFoundError,
)


class TestDatabeakError:
    """Test base DatabeakError class."""

    def test_basic_error_creation(self) -> None:
        """Test basic error creation."""
        error = DatabeakError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"

    def test_error_inheritance(self) -> None:
        """Test that DatabeakError inherits from Exception."""
        error = DatabeakError("Test error")
        assert isinstance(error, Exception)

    def test_error_with_details(self) -> None:
        """Test error with details and error code."""
        error = DatabeakError("Error", error_code="TEST_ERROR", details={"key": "value"})
        assert error.message == "Error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}

    def test_error_to_dict(self) -> None:
        """Test error serialization to dictionary."""
        error = DatabeakError("Error message", error_code="TEST_CODE", details={"key": "value"})
        error_dict = error.to_dict()

        assert error_dict["type"] == "DatabeakError"
        assert error_dict["message"] == "Error message"
        assert error_dict["error_code"] == "TEST_CODE"
        assert error_dict["details"] == {"key": "value"}

    def test_error_to_dict_without_code(self) -> None:
        """Test error serialization without error code."""
        error = DatabeakError("Simple error")
        error_dict = error.to_dict()

        assert error_dict["type"] == "DatabeakError"
        assert error_dict["message"] == "Simple error"
        assert error_dict["error_code"] is None
        assert error_dict["details"] == {}


class TestSessionNotFoundError:
    """Test SessionNotFoundError class."""

    def test_session_not_found_basic(self) -> None:
        """Test basic session not found error."""
        error = SessionNotFoundError("session_123")
        assert "session_123" in str(error)
        assert isinstance(error, DatabeakError)

    def test_session_not_found_inheritance(self) -> None:
        """Test inheritance chain."""
        error = SessionNotFoundError("test_session")
        assert isinstance(error, DatabeakError)
        assert isinstance(error, Exception)


class TestColumnNotFoundError:
    """Test ColumnNotFoundError class."""

    def test_column_not_found_basic(self) -> None:
        """Test basic column not found error."""
        error = ColumnNotFoundError("age", ["name", "email", "city"])
        error_str = str(error)
        assert "age" in error_str

    def test_column_not_found_inheritance(self) -> None:
        """Test inheritance chain."""
        error = ColumnNotFoundError("test_col", ["other_col"])
        assert isinstance(error, DatabeakError)
        assert isinstance(error, Exception)


class TestOperationError:
    """Test OperationError class."""

    def test_operation_error_basic(self) -> None:
        """Test basic operation error."""
        error = OperationError("Operation failed")
        assert str(error) == "Operation failed"
        assert isinstance(error, DatabeakError)

    def test_operation_error_inheritance(self) -> None:
        """Test inheritance chain."""
        error = OperationError("Test operation error")
        assert isinstance(error, DatabeakError)
        assert isinstance(error, Exception)


class TestExceptionChaining:
    """Test exception chaining and cause handling."""

    def test_exception_chaining(self) -> None:
        """Test that exceptions can be chained properly."""

        # Test exception chaining using pytest.raises
        def raise_chained_error() -> None:
            try:
                msg = "Original error"
                raise ValueError(msg)  # noqa: TRY301
            except ValueError as e:
                msg = "DataBeak error"
                raise DatabeakError(msg) from e

        with pytest.raises(DatabeakError) as exc_info:
            raise_chained_error()

        # Verify chaining worked correctly
        chained_error = exc_info.value
        assert chained_error.__cause__ is not None
        assert isinstance(chained_error.__cause__, ValueError)

    def test_nested_exception_handling(self) -> None:
        """Test nested exception scenarios."""

        # Test nested exception handling using pytest.raises
        def raise_nested_error() -> None:
            try:
                msg = "missing_col"
                raise ColumnNotFoundError(msg, ["available_col"])  # noqa: TRY301
            except ColumnNotFoundError as col_error:
                msg = "Operation failed due to column issue"
                raise OperationError(msg) from col_error

        with pytest.raises(OperationError) as exc_info:
            raise_nested_error()

        # Verify nested exception structure
        op_error = exc_info.value
        assert isinstance(op_error.__cause__, ColumnNotFoundError)
        assert "missing_col" in str(op_error.__cause__)


class TestSessionExpiredError:
    """Test SessionExpiredError class."""

    def test_session_expired_basic(self) -> None:
        """Test basic session expired error."""
        error = SessionExpiredError("session_456")
        assert "session_456" in str(error)
        assert "expired" in str(error).lower()
        assert error.error_code == "SESSION_EXPIRED"
        assert error.details["session_id"] == "session_456"


class TestInvalidRowIndexError:
    """Test InvalidRowIndexError class."""

    def test_invalid_row_index_basic(self) -> None:
        """Test basic invalid row index error."""
        error = InvalidRowIndexError(100, 50)
        assert "100" in str(error)
        assert error.error_code == "INVALID_ROW_INDEX"
        assert error.details["row_index"] == 100
        assert error.details["max_index"] == 50

    def test_invalid_row_index_without_max(self) -> None:
        """Test invalid row index error without max index."""
        error = InvalidRowIndexError(100)
        assert "100" in str(error)
        assert error.details["row_index"] == 100
        assert error.details["max_index"] is None


class TestFileErrors:
    """Test file-related error classes."""

    def test_file_not_found_error(self) -> None:
        """Test DataBeakFileNotFoundError."""
        error = DataBeakFileNotFoundError("/path/to/missing.csv")
        assert "/path/to/missing.csv" in str(error)
        assert error.error_code == "FILE_NOT_FOUND"
        assert error.details["file_path"] == "/path/to/missing.csv"

    def test_file_permission_error(self) -> None:
        """Test FilePermissionError."""
        error = FilePermissionError("/path/to/file.csv", "read")
        assert "/path/to/file.csv" in str(error)
        assert "read" in str(error)
        assert error.error_code == "FILE_PERMISSION_ERROR"
        assert error.details["file_path"] == "/path/to/file.csv"
        assert error.details["operation"] == "read"

    def test_file_format_error(self) -> None:
        """Test FileFormatError."""
        error = FileFormatError("/path/to/file.txt", "CSV")
        assert "/path/to/file.txt" in str(error)
        assert error.error_code == "FILE_FORMAT_ERROR"
        assert error.details["file_path"] == "/path/to/file.txt"
        assert error.details["expected_format"] == "CSV"

    def test_file_format_error_without_expected(self) -> None:
        """Test FileFormatError without expected format."""
        error = FileFormatError("/path/to/file.txt")
        assert "/path/to/file.txt" in str(error)
        assert error.details["expected_format"] is None


class TestOperationErrors:
    """Test operation-related error classes."""

    def test_invalid_operation_error(self) -> None:
        """Test InvalidOperationError."""
        error = InvalidOperationError("sort", "no data loaded")
        assert "sort" in str(error)
        assert "no data loaded" in str(error)
        assert error.error_code == "INVALID_OPERATION"
        assert error.details["operation"] == "sort"
        assert error.details["reason"] == "no data loaded"


class TestParameterErrors:
    """Test parameter-related error classes."""

    def test_invalid_parameter_error(self) -> None:
        """Test InvalidParameterError."""
        error = InvalidParameterError("column_name", "invalid$name", "alphanumeric characters")
        assert "column_name" in str(error)
        assert "invalid$name" in str(error)
        assert error.error_code == "INVALID_PARAMETER"
        assert error.details["parameter"] == "column_name"
        assert error.details["value"] == "invalid$name"
        assert error.details["expected"] == "alphanumeric characters"

    def test_invalid_parameter_error_without_expected(self) -> None:
        """Test InvalidParameterError without expected description."""
        error = InvalidParameterError("limit", -5)
        assert "limit" in str(error)
        assert error.details["parameter"] == "limit"
        assert error.details["value"] == "-5"
        assert error.details["expected"] is None

    def test_missing_parameter_error(self) -> None:
        """Test MissingParameterError."""
        error = MissingParameterError("session_id")
        assert "session_id" in str(error)
        assert "missing" in str(error).lower()
        assert error.error_code == "MISSING_PARAMETER"
        assert error.details["parameter"] == "session_id"


class TestErrorMessageFormatting:
    """Test error message formatting and readability."""

    def test_column_not_found_message_format(self) -> None:
        """Test column not found message is user-friendly."""
        available_cols = ["name", "age", "email", "city"]
        error = ColumnNotFoundError("missing_column", available_cols)
        error_msg = str(error)

        # Should contain the missing column
        assert "missing_column" in error_msg

    def test_session_not_found_message_format(self) -> None:
        """Test session not found message is informative."""
        error = SessionNotFoundError("session_123")
        error_msg = str(error)

        assert "session_123" in error_msg
        assert len(error_msg) > len("session_123")  # Should have descriptive text

    def test_error_messages_are_strings(self) -> None:
        """Test that all error messages are proper strings."""
        errors = [
            DatabeakError("Test error"),
            SessionNotFoundError("test_session"),
            ColumnNotFoundError("col", ["other"]),
            OperationError("Operation failed"),
        ]

        for error in errors:
            error_str = str(error)
            assert isinstance(error_str, str)
            assert len(error_str) > 0
            assert error_str != "None"
