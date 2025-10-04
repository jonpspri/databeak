"""Security tests for fixture path handling."""

import pytest

from tests.integration.conftest import get_fixture_path


class TestFixtureSecurity:
    """Test security validation in fixture path handling."""

    def test_valid_fixture_names(self) -> None:
        """Test that valid fixture names work correctly."""
        # These should work without raising exceptions
        path1 = get_fixture_path("sample_data.csv")
        assert path1.endswith("sample_data.csv")
        assert "tests/fixtures" in path1

        path2 = get_fixture_path("test_file.json")
        assert path2.endswith("test_file.json")
        assert "tests/fixtures" in path2

    def test_reject_path_separators(self) -> None:
        """Test that fixture names with path separators are rejected."""
        # Unix path separator
        with pytest.raises(ValueError, match="cannot contain path separators"):
            get_fixture_path("../etc/passwd")

        with pytest.raises(ValueError, match="cannot contain path separators"):
            get_fixture_path("subdir/file.csv")

        with pytest.raises(ValueError, match="cannot contain path separators"):
            get_fixture_path("/absolute/path.csv")

        # Windows path separator (if different from Unix)
        import os

        if os.path.altsep:
            with pytest.raises(ValueError, match="cannot contain path separators"):
                get_fixture_path(f"subdir{os.path.altsep}file.csv")

    def test_reject_relative_path_components(self) -> None:
        """Test that relative path components are rejected."""
        # Note: ../sensitive_file.csv is caught by path separator check first
        with pytest.raises(ValueError, match="path separators|relative path components"):
            get_fixture_path("../sensitive_file.csv")

        with pytest.raises(ValueError, match="cannot contain relative path components"):
            get_fixture_path("..something.csv")

        with pytest.raises(ValueError, match="cannot contain relative path components"):
            get_fixture_path(".hidden_file")

        with pytest.raises(ValueError, match="cannot contain relative path components"):
            get_fixture_path(".env")

    def test_directory_traversal_attempts(self) -> None:
        """Test various directory traversal attack patterns."""
        traversal_attempts = [
            "../../etc/passwd",
            "../../../root/.ssh/id_rsa",
            "....//....//etc/passwd",
            "..\\..\\.\\windows\\system32\\config\\sam",  # Windows style
            "..%2f..%2fetc%2fpasswd",  # URL encoded (though this might not be relevant here)
            "..\\..\\etc\\passwd",  # Mixed separators
        ]

        for attempt in traversal_attempts:
            with pytest.raises(
                ValueError, match="path separators|relative path components"
            ) as exc_info:
                get_fixture_path(attempt)

            # Should be caught by one of our validations
            assert "path separators" in str(exc_info.value) or "relative path components" in str(
                exc_info.value
            )

    def test_path_stays_within_fixtures_directory(self) -> None:
        """Test that resolved paths stay within fixtures directory."""
        # This test assumes the validation logic works correctly
        # If someone found a way to bypass the separator checks,
        # the final directory containment check should catch it

        valid_path = get_fixture_path("sample_data.csv")

        # The resolved path should be within fixtures directory
        assert "tests/fixtures" in valid_path

        # Should be an absolute path
        assert valid_path.startswith("/")

    def test_empty_and_whitespace_names(self) -> None:
        """Test handling of empty and whitespace-only fixture names."""
        # Empty string
        with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
            get_fixture_path("")

        # Just whitespace
        with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
            get_fixture_path("   ")

        # Just dots
        with pytest.raises(ValueError, match="cannot contain relative path components"):
            get_fixture_path("...")

    def test_very_long_names(self) -> None:
        """Test handling of very long fixture names."""
        # Very long name (but otherwise valid)
        long_name = "a" * 200 + ".csv"

        # Should work (though file won't exist)
        path = get_fixture_path(long_name)
        assert path.endswith(long_name)

    def test_special_characters_allowed(self) -> None:
        """Test that some special characters are allowed in fixture names."""
        # These should be allowed
        valid_special_names = [
            "file-with-dashes.csv",
            "file_with_underscores.csv",
            "file.with.dots.csv",
            "file@symbol.csv",
            "file#hash.csv",
            "file with spaces.csv",
        ]

        for name in valid_special_names:
            path = get_fixture_path(name)
            assert path.endswith(name)
            assert "tests/fixtures" in path
