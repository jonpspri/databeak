"""Unit tests for csv_session.py module."""

from src.databeak.models.csv_session import DataBeakSettings, get_session_manager


class TestDataBeakSettings:
    """Tests for DataBeakSettings configuration."""

    def test_default_settings(self):
        """Test default settings initialization."""
        settings = DataBeakSettings()
        assert settings.auto_save is True
        assert settings.session_timeout == 3600
        assert settings.csv_history_dir == "."
        assert settings.max_file_size_mb == 1024


class TestSessionManager:
    """Tests for session manager functionality."""

    def test_get_session_manager(self):
        """Test getting session manager instance."""
        manager = get_session_manager()
        assert manager is not None
        # Singleton pattern
        manager2 = get_session_manager()
        assert manager is manager2
