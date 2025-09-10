"""Unit tests for csv_session.py module."""

from src.databeak.models.csv_session import DataBeakSettings, get_session_manager


class TestDataBeakSettings:
    """Tests for DataBeakSettings configuration."""

    def test_default_settings(self):
        """Test default settings initialization."""
        settings = DataBeakSettings()
        assert settings.AUTO_SAVE_ENABLED is True
        assert settings.AUTO_SAVE_INTERVAL_SECONDS == 30
        assert settings.SESSION_TIMEOUT_SECONDS == 3600
        assert settings.MAX_HISTORY_SIZE == 100


class TestSessionManager:
    """Tests for session manager functionality."""

    def test_get_session_manager(self):
        """Test getting session manager instance."""
        manager = get_session_manager()
        assert manager is not None
        # Singleton pattern
        manager2 = get_session_manager()
        assert manager is manager2
