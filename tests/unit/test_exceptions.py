"""Unit tests for exception classes."""

import pytest

from databeak.exceptions import (
    ColumnNotFoundError,
    DatabeakError,
    OperationError,
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
                raise ValueError(msg) # noqa: TRY301
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
                raise ColumnNotFoundError(msg, ["available_col"]) # noqa: TRY301
            except ColumnNotFoundError as col_error:
                msg = "Operation failed due to column issue"
                raise OperationError(msg) from col_error

        with pytest.raises(OperationError) as exc_info:
            raise_nested_error()

        # Verify nested exception structure
        op_error = exc_info.value
        assert isinstance(op_error.__cause__, ColumnNotFoundError)
        assert "missing_col" in str(op_error.__cause__)


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
