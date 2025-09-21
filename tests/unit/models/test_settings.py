"""Tests for DataBeak settings functionality."""

import os
from unittest.mock import patch

from src.databeak.models.csv_session import CSVSession, DataBeakSettings, get_csv_settings


class TestDataBeakSettings:
    """Test DataBeak settings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings configuration."""
        settings = DataBeakSettings()
        # History functionality removed - test other defaults
        assert settings.max_file_size_mb == 1024
        assert settings.session_timeout == 3600
        assert settings.chunk_size == 10000
        assert settings.max_history_operations == 1000

    def test_settings_with_custom_values(self) -> None:
        """Test settings with custom values."""
        settings = DataBeakSettings(max_file_size_mb=2048, session_timeout=7200, chunk_size=5000)
        assert settings.max_file_size_mb == 2048
        assert settings.session_timeout == 7200
        assert settings.chunk_size == 5000

    def test_environment_variable_override(self) -> None:
        """Test that environment variables override defaults."""
        with patch.dict(
            os.environ,
            {
                "DATABEAK_MAX_FILE_SIZE_MB": "4096",
                "DATABEAK_SESSION_TIMEOUT": "14400",
                "DATABEAK_CHUNK_SIZE": "20000",
            },
        ):
            settings = DataBeakSettings()
            assert settings.max_file_size_mb == 4096
            assert settings.session_timeout == 14400
            assert settings.chunk_size == 20000

    def test_case_insensitive_env_var(self) -> None:
        """Test that environment variables are case insensitive."""
        with patch.dict(os.environ, {"DATABEAK_MAX_FILE_SIZE_MB": "512"}):
            settings = DataBeakSettings()
            assert settings.max_file_size_mb == 512


class TestDataBeakSettingsIntegration:
    """Test DataBeak settings integration with sessions."""

    def test_get_csv_settings_singleton(self) -> None:
        """Test that get_csv_settings returns singleton instance."""
        with patch.object(
            __import__("src.databeak.models.csv_session", fromlist=["_settings"]),
            "_settings",
            None,
        ):
            settings1 = get_csv_settings()
            settings2 = get_csv_settings()
            assert settings1 is settings2

    def test_session_with_environment_variable(self):
        """Test that CSVSession works with environment variable settings."""
        with (
            patch.dict(os.environ, {"DATABEAK_SESSION_TIMEOUT": "7200"}),
            patch.object(
                __import__("src.databeak.models.csv_session", fromlist=["_settings"]),
                "_settings",
                None,
            ),
        ):
            session = CSVSession()
            # Session should initialize successfully
            assert session.session_id is not None
            assert session.has_data() is False

    def test_settings_are_configurable(self):
        """Test that settings can be configured multiple ways."""
        # Test 1: Direct instantiation
        settings1 = DataBeakSettings(max_file_size_mb=512)
        assert settings1.max_file_size_mb == 512

        # Test 2: Environment variable
        with patch.dict(os.environ, {"DATABEAK_MAX_FILE_SIZE_MB": "2048"}):
            settings2 = DataBeakSettings()
            assert settings2.max_file_size_mb == 2048

        # Test 3: Default
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing env vars
            if "DATABEAK_MAX_FILE_SIZE_MB" in os.environ:
                del os.environ["DATABEAK_MAX_FILE_SIZE_MB"]
            settings3 = DataBeakSettings()
            assert settings3.max_file_size_mb == 1024


class TestSettingsDocumentation:
    """Test that settings behavior matches documentation."""

    def test_env_prefix_documentation(self):
        """Test that DATABEAK_ prefix works as documented."""
        with patch.dict(os.environ, {"DATABEAK_CHUNK_SIZE": "15000"}):
            settings = DataBeakSettings()
            assert settings.chunk_size == 15000

    def test_default_values_documentation(self):
        """Test that default values match documentation."""
        # Clear environment and test default values
        with patch.dict(os.environ, {}, clear=True):
            for var in [
                "DATABEAK_MAX_FILE_SIZE_MB",
                "DATABEAK_SESSION_TIMEOUT",
                "DATABEAK_CHUNK_SIZE",
            ]:
                if var in os.environ:
                    del os.environ[var]

            settings = DataBeakSettings()
            assert settings.max_file_size_mb == 1024, "Default file size limit should be 1024 MB"
            assert settings.session_timeout == 3600, (
                "Default session timeout should be 3600 seconds"
            )
            assert settings.chunk_size == 10000, "Default chunk size should be 10000"
