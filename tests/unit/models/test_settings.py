"""Tests for DataBeak settings functionality."""

import os
from unittest.mock import patch

from databeak.core.settings import DataBeakSettings


class TestDataBeakSettings:
    """Test DataBeak settings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings configuration."""
        settings = DataBeakSettings()
        # History functionality removed - test other defaults
        assert settings.max_url_size_mb == 100
        assert settings.session_timeout == 3600
        assert settings.max_anomaly_sample_size == 10000

    def test_settings_with_custom_values(self) -> None:
        """Test settings with custom values."""
        settings = DataBeakSettings(max_url_size_mb=200, session_timeout=7200)
        assert settings.max_url_size_mb == 200
        assert settings.session_timeout == 7200

    def test_environment_variable_override(self) -> None:
        """Test that environment variables override defaults."""
        with patch.dict(
            os.environ,
            {
                "DATABEAK_MAX_URL_SIZE_MB": "200",
                "DATABEAK_SESSION_TIMEOUT": "14400",
            },
        ):
            settings = DataBeakSettings()
            assert settings.max_url_size_mb == 200
            assert settings.session_timeout == 14400

    def test_case_insensitive_env_var(self) -> None:
        """Test that environment variables are case insensitive."""
        with patch.dict(os.environ, {"DATABEAK_MAX_URL_SIZE_MB": "512"}):
            settings = DataBeakSettings()
            assert settings.max_url_size_mb == 512


class TestDataBeakSettingsIntegration:
    """Test DataBeak settings integration with sessions."""

    def test_settings_are_configurable(self) -> None:
        """Test that settings can be configured multiple ways."""
        # Test 1: Direct instantiation
        settings1 = DataBeakSettings(max_url_size_mb=512)
        assert settings1.max_url_size_mb == 512

        # Test 2: Environment variable
        with patch.dict(os.environ, {"DATABEAK_MAX_URL_SIZE_MB": "2048"}):
            settings2 = DataBeakSettings()
            assert settings2.max_url_size_mb == 2048

        # Test 3: Default
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing env vars
            if "DATABEAK_MAX_URL_SIZE_MB" in os.environ:
                del os.environ["DATABEAK_MAX_URL_SIZE_MB"]
            settings3 = DataBeakSettings()
            assert settings3.max_url_size_mb == 100


class TestSettingsDocumentation:
    """Test that settings behavior matches documentation."""

    def test_env_prefix_documentation(self) -> None:
        """Test that DATABEAK_ prefix works as documented."""
        with patch.dict(os.environ, {"DATABEAK_URL_TIMEOUT_SECONDS": "60"}):
            settings = DataBeakSettings()
            assert settings.url_timeout_seconds == 60

    def test_default_values_documentation(self) -> None:
        """Test that default values match documentation."""
        # Clear environment and test default values
        with patch.dict(os.environ, {}, clear=True):
            for var in [
                "DATABEAK_MAX_URL_SIZE_MB",
                "DATABEAK_SESSION_TIMEOUT",
            ]:
                if var in os.environ:
                    del os.environ[var]

            settings = DataBeakSettings()
            assert settings.max_url_size_mb == 100, "Default URL size limit should be 100 MB"
            assert settings.session_timeout == 3600, (
                "Default session timeout should be 3600 seconds"
            )
