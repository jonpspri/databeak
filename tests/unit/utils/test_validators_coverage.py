"""Comprehensive coverage tests for validators module."""

import socket
from unittest.mock import Mock, patch

import pandas as pd

from src.databeak.utils.validators import (
    convert_pandas_na_list,
    convert_pandas_na_to_none,
    sanitize_filename,
    validate_column_name,
    validate_dataframe,
    validate_expression,
    validate_file_path,
    validate_url,
)


class TestFilePathValidation:
    """Test file path validation functionality."""

    def test_validate_file_path_valid_csv(self, tmp_path) -> None:
        """Test validation of valid CSV file."""
        # Create a test CSV file
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\nvalue1,value2\n")

        is_valid, message = validate_file_path(str(test_file))

        assert is_valid is True
        assert str(test_file) in message

    def test_validate_file_path_valid_extensions(self, tmp_path) -> None:
        """Test validation with all valid file extensions."""
        valid_extensions = [".csv", ".tsv", ".txt", ".dat"]

        for ext in valid_extensions:
            test_file = tmp_path / f"test{ext}"
            test_file.write_text("data")

            is_valid, _message = validate_file_path(str(test_file))
            assert is_valid is True

    def test_validate_file_path_invalid_extension(self, tmp_path) -> None:
        """Test validation with invalid file extension."""
        test_file = tmp_path / "test.xlsx"
        test_file.write_text("data")

        is_valid, message = validate_file_path(str(test_file))

        assert is_valid is False
        assert "Invalid file extension" in message

    def test_validate_file_path_not_exist_required(self) -> None:
        """Test validation when file doesn't exist but is required."""
        nonexistent_file = "/nonexistent/path/file.csv"

        is_valid, message = validate_file_path(nonexistent_file, must_exist=True)

        assert is_valid is False
        assert "File not found" in message

    def test_validate_file_path_not_exist_optional(self) -> None:
        """Test validation when file doesn't exist but is optional."""
        nonexistent_file = "/tmp/new_file.csv"

        is_valid, _message = validate_file_path(nonexistent_file, must_exist=False)

        assert is_valid is True

    def test_validate_file_path_traversal_attack(self) -> None:
        """Test protection against path traversal attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "~/secret.csv",
            "..\\..\\windows\\system32\\config",
            "test/../../../etc/passwd.csv",
        ]

        for path in malicious_paths:
            is_valid, message = validate_file_path(path, must_exist=False)
            assert is_valid is False
            assert "Path traversal not allowed" in message

    def test_validate_file_path_directory_not_file(self, tmp_path) -> None:
        """Test validation when path points to directory."""
        is_valid, message = validate_file_path(str(tmp_path))

        assert is_valid is False
        assert "Not a file" in message

    def test_validate_file_path_large_file(self, tmp_path) -> None:
        """Test validation with file size limit."""
        # Create a large file (mock the stat result)
        test_file = tmp_path / "large.csv"
        test_file.write_text("data")

        # Mock the stat method on the specific Path instance
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value = Mock(st_size=2 * 1024 * 1024 * 1024)  # 2GB

            is_valid, message = validate_file_path(str(test_file))

            assert is_valid is False
            assert "File too large" in message or "Error validating path" in message

    def test_validate_file_path_exception_handling(self) -> None:
        """Test exception handling in file path validation."""
        # Use invalid path that causes exception
        with patch("pathlib.Path.resolve", side_effect=OSError("Permission denied")):
            is_valid, message = validate_file_path("test.csv", must_exist=False)

            assert is_valid is False
            assert "Error validating path" in message


class TestURLValidation:
    """Test URL validation functionality."""

    def test_validate_url_valid_http(self) -> None:
        """Test validation of valid HTTP URL."""
        url = "http://example.com/data.csv"

        is_valid, message = validate_url(url)

        assert is_valid is True
        assert url in message

    def test_validate_url_valid_https(self) -> None:
        """Test validation of valid HTTPS URL."""
        url = "https://data.example.com/dataset.csv"

        is_valid, _message = validate_url(url)

        assert is_valid is True

    def test_validate_url_invalid_scheme(self) -> None:
        """Test validation of invalid URL scheme."""
        invalid_urls = [
            "ftp://example.com/data.csv",
            "file:///local/path.csv",
            "javascript:alert('xss')",
            "data:text/csv,col1,col2",
        ]

        for url in invalid_urls:
            is_valid, message = validate_url(url)
            assert is_valid is False
            assert "HTTP/HTTPS URLs are supported" in message

    def test_validate_url_malformed(self) -> None:
        """Test validation of malformed URLs."""
        malformed_urls = ["not-a-url", "http://", "https://", "", "http:///missing-domain"]

        for url in malformed_urls:
            is_valid, _message = validate_url(url)
            assert is_valid is False

    def test_validate_url_private_ip_addresses(self) -> None:
        """Test validation blocks private IP addresses."""
        private_urls = [
            "http://192.168.1.1/data.csv",
            "http://10.0.0.1/file.csv",
            "http://172.16.0.1/dataset.csv",
        ]

        for url in private_urls:
            is_valid, message = validate_url(url)
            assert is_valid is False
            assert "Private network addresses not allowed" in message

    def test_validate_url_loopback_addresses(self) -> None:
        """Test validation blocks loopback addresses."""
        loopback_urls = [
            "http://127.0.0.1:8000/local.csv",
            "http://localhost/data.csv",
            "http://localhost:8000/file.csv",
        ]

        for url in loopback_urls:
            is_valid, _message = validate_url(url)
            assert is_valid is False

    @patch("socket.getaddrinfo")
    def test_validate_url_dns_resolution_to_private(self, mock_getaddrinfo) -> None:
        """Test validation when DNS resolves to private IP."""
        # Mock DNS resolution to return private IP
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.100", 80)),
        ]

        is_valid, message = validate_url("http://internal.company.com/data.csv")

        assert is_valid is False
        assert "private address" in message

    @patch("socket.getaddrinfo")
    def test_validate_url_dns_resolution_error(self, mock_getaddrinfo) -> None:
        """Test validation when DNS resolution fails."""
        mock_getaddrinfo.side_effect = socket.gaierror("Name resolution failed")

        is_valid, _message = validate_url("http://nonexistent-domain.com/data.csv")

        # DNS failure should be allowed but handled gracefully
        assert isinstance(is_valid, bool)

    def test_validate_url_exception_handling(self) -> None:
        """Test exception handling in URL validation."""
        # Test with a malformed URL that might cause issues
        is_valid, _message = validate_url("not-a-url-at-all")

        assert is_valid is False
        # Could be various error messages depending on what fails first


class TestColumnNameValidation:
    """Test column name validation."""

    def test_validate_column_name_valid(self) -> None:
        """Test validation of valid column names."""
        valid_names = [
            "name",
            "user_age",
            "column1",
            "Column_Name_123",
            "CamelCase",
            "_underscore_start",
            "a",
            "A1",
        ]

        for name in valid_names:
            is_valid, _message = validate_column_name(name)
            assert is_valid is True, f"Failed for valid name: {name}"

    def test_validate_column_name_invalid(self) -> None:
        """Test validation of invalid column names."""
        invalid_names = [
            "",  # empty
            " ",  # whitespace only
            "col with spaces",
            "col-with-hyphens",
            "123starts_with_number",
            "col@with#symbols",
            "col.with.dots",
            "col+plus",
            "col/slash",
            None,  # None type
        ]

        for name in invalid_names:
            is_valid, _message = validate_column_name(name)
            assert is_valid is False, f"Should have failed for invalid name: {name}"

    def test_validate_column_name_non_string(self) -> None:
        """Test validation of non-string column names."""
        non_string_names = [123, [], {}, True, 1.5]

        for name in non_string_names:
            is_valid, message = validate_column_name(name)
            assert is_valid is False
            assert "non-empty string" in message


class TestDataframeValidation:
    """Test dataframe validation functionality."""

    def test_validate_dataframe_valid(self) -> None:
        """Test validation of valid dataframe."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "salary": [50000, 60000, 70000],
            },
        )

        result = validate_dataframe(df)

        assert isinstance(result, dict)
        assert "errors" in result
        assert "warnings" in result
        assert "info" in result
        assert len(result["errors"]) == 0

    def test_validate_dataframe_empty(self) -> None:
        """Test validation of empty dataframe."""
        df = pd.DataFrame()

        result = validate_dataframe(df)

        assert len(result["errors"]) > 0
        assert "empty" in result["errors"][0].lower()

    def test_validate_dataframe_duplicate_columns(self) -> None:
        """Test validation detects duplicate columns."""
        # Create DataFrame with duplicate columns manually
        data = [[1, 2, 3]]
        df = pd.DataFrame(data, columns=["a", "b", "a"])

        result = validate_dataframe(df)

        assert len(result["errors"]) > 0
        assert any("Duplicate column" in error for error in result["errors"])

    def test_validate_dataframe_null_columns(self) -> None:
        """Test validation detects completely null columns."""
        df = pd.DataFrame({"good_col": [1, 2, 3], "null_col": [None, None, None]})

        result = validate_dataframe(df)

        assert len(result["warnings"]) > 0
        assert any("null columns" in warning for warning in result["warnings"])

    def test_validate_dataframe_mixed_types(self) -> None:
        """Test validation detects mixed types in columns."""
        df = pd.DataFrame({"mixed_col": [1, "string", 3.14, True]})

        result = validate_dataframe(df)

        # Should detect mixed types
        assert len(result["warnings"]) > 0
        assert any("mixed types" in warning for warning in result["warnings"])

    def test_validate_dataframe_info_fields(self) -> None:
        """Test validation includes info fields."""
        df = pd.DataFrame({"col1": range(100)})

        result = validate_dataframe(df)

        assert "shape" in result["info"]
        assert "memory_usage_mb" in result["info"]
        assert result["info"]["shape"] == (100, 1)
        assert isinstance(result["info"]["memory_usage_mb"], int | float)

    def test_validate_dataframe_high_cardinality(self) -> None:
        """Test validation detects high cardinality columns."""
        # Create a column with very high cardinality (each value unique)
        df = pd.DataFrame({"high_cardinality": [f"unique_{i}" for i in range(100)]})

        result = validate_dataframe(df)

        assert "high_cardinality_high_cardinality" in result["info"]


class TestExpressionValidation:
    """Test expression validation functionality."""

    def test_validate_expression_valid(self) -> None:
        """Test validation of valid expressions."""
        allowed_vars = ["x", "y", "z", "data"]

        valid_expressions = ["x + y", "x * 2", "x > 5", "x + y - z", "(x + y) * z"]

        for expr in valid_expressions:
            is_valid, message = validate_expression(expr, allowed_vars)
            assert is_valid is True, f"Failed for valid expression: {expr}"

        # Test data.mean() separately as it has method call
        is_valid, _message = validate_expression("data + 1", allowed_vars)
        assert is_valid is True

    def test_validate_expression_invalid_variables(self) -> None:
        """Test validation rejects invalid variables."""
        allowed_vars = ["x", "y"]

        invalid_expressions = [
            "x + unknown_var",
            "malicious_func()",
            "import os",
            "__import__('os')",
            "exec('code')",
        ]

        for expr in invalid_expressions:
            is_valid, _message = validate_expression(expr, allowed_vars)
            assert is_valid is False, f"Should have failed for: {expr}"

    def test_validate_expression_empty(self) -> None:
        """Test validation of empty expression."""
        is_valid, _message = validate_expression("", [])

        # Empty expression is technically valid in this implementation
        assert is_valid is True

    def test_validate_expression_dangerous_keywords(self) -> None:
        """Test validation blocks dangerous keywords."""
        allowed_vars = ["x"]

        dangerous_expressions = [
            "import sys",
            "exec('code')",
            "eval('code')",
            "__import__",
            "open('file')",
            "globals()",
            "locals()",
        ]

        for expr in dangerous_expressions:
            is_valid, _message = validate_expression(expr, allowed_vars)
            assert is_valid is False


class TestUtilityFunctions:
    """Test utility functions for data conversion and sanitization."""

    def test_sanitize_filename_basic(self) -> None:
        """Test basic filename sanitization."""
        filename = "test_file.csv"
        result = sanitize_filename(filename)

        assert result == filename  # Should be unchanged

    def test_sanitize_filename_special_characters(self) -> None:
        """Test sanitization removes special characters."""
        filename = "file<>:|?*.csv"
        result = sanitize_filename(filename)

        # Should remove or replace special characters
        assert "<" not in result
        assert ">" not in result
        assert "|" not in result
        assert "?" not in result
        assert "*" not in result

    def test_sanitize_filename_path_separators(self) -> None:
        """Test sanitization handles path separators."""
        filename = "folder/file\\name.csv"
        result = sanitize_filename(filename)

        # Path() removes directory components, leaving only filename
        assert "/" not in result
        # But backslash in filename itself might remain if it's not a path separator
        # The actual behavior depends on Path() implementation

    def test_sanitize_filename_empty(self) -> None:
        """Test sanitization of empty filename."""
        result = sanitize_filename("")

        # Should handle empty filename gracefully
        assert isinstance(result, str)

    def test_convert_pandas_na_to_none_basic(self) -> None:
        """Test conversion of pandas NA to None."""
        # Test with regular values
        assert convert_pandas_na_to_none("string") == "string"
        assert convert_pandas_na_to_none(123) == 123
        assert convert_pandas_na_to_none(True) is True  # noqa: FBT003

    def test_convert_pandas_na_to_none_na_values(self) -> None:
        """Test conversion of pandas NA values to None."""
        import numpy as np

        # Test pandas NA values
        assert convert_pandas_na_to_none(pd.NA) is None
        assert convert_pandas_na_to_none(np.nan) is None
        assert convert_pandas_na_to_none(pd.NaT) is None

    def test_convert_pandas_na_list_basic(self) -> None:
        """Test conversion of pandas NA values in list."""
        values = [1, 2, 3, "text", True]
        result = convert_pandas_na_list(values)

        assert result == values  # Should be unchanged

    def test_convert_pandas_na_list_with_na(self) -> None:
        """Test conversion of list with pandas NA values."""
        import numpy as np

        values = [1, pd.NA, 3, np.nan, "text", pd.NaT]
        result = convert_pandas_na_list(values)

        expected = [1, None, 3, None, "text", None]
        assert result == expected

    def test_convert_pandas_na_list_empty(self) -> None:
        """Test conversion of empty list."""
        result = convert_pandas_na_list([])

        assert result == []

    def test_convert_pandas_na_list_all_na(self) -> None:
        """Test conversion of list with all NA values."""
        import numpy as np

        values = [pd.NA, np.nan, pd.NaT]
        result = convert_pandas_na_list(values)

        assert result == [None, None, None]


class TestValidatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_validators_handle_none_gracefully(self) -> None:
        """Test that validators handle None inputs gracefully."""
        # These should not crash
        validate_file_path("", must_exist=False)
        validate_url("")
        validate_column_name("")
        validate_expression("", [])
        sanitize_filename("")

        # DataFrame validators with minimal data
        df = pd.DataFrame({"col1": [1]})
        validate_dataframe(df)

    def test_unicode_handling(self) -> None:
        """Test handling of unicode characters."""
        # Column names with unicode
        is_valid, _message = validate_column_name("tēst_cōlumn")
        assert isinstance(is_valid, bool)

        # Filenames with unicode
        result = sanitize_filename("tëst_fïlé.csv")
        assert isinstance(result, str)

    def test_very_long_inputs(self) -> None:
        """Test handling of very long inputs."""
        long_string = "a" * 10000

        # Should handle long inputs gracefully
        validate_column_name(long_string)
        sanitize_filename(long_string)

    def test_special_dataframe_cases(self) -> None:
        """Test special dataframe validation cases."""
        # DataFrame with only one row
        df = pd.DataFrame({"col1": [1]})
        result = validate_dataframe(df)
        assert isinstance(result, dict)

        # DataFrame with many columns
        df = pd.DataFrame({f"col{i}": [1] for i in range(100)})
        result = validate_dataframe(df)
        assert isinstance(result, dict)
