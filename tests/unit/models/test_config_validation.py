"""Test configuration validation to ensure settings work correctly."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

from src.databeak.models.csv_session import DataBeakSettings


class TestVersionLoading:
    """Test version loading functionality."""

    def test_package_version_loading(self):
        """Test that version can be loaded from package metadata."""
        # This should work in both development and production
        version = importlib.metadata.version("databeak")
        assert version is not None
        assert isinstance(version, str)
        assert len(version.strip()) > 0

    def test_version_module_import(self):
        """Test that version module imports and provides version."""
        from src.databeak._version import VERSION, __version__

        assert __version__ is not None
        assert VERSION is not None
        assert __version__ == VERSION
        assert isinstance(__version__, str)

    def test_version_fallback_mechanism(self):
        """Test that version fallback mechanism is in place."""
        from src.databeak._version import __version__

        # Version should be either from metadata or fallback
        assert __version__ is not None
        # Should be either real version or dev fallback
        assert __version__ != ""


class TestEnvironmentVariableConfiguration:
    """Test environment variable configuration matches DataBeakSettings."""

    def test_databeak_settings_has_correct_prefix(self):
        """Test that DataBeakSettings uses DATABEAK_ prefix."""
        settings = DataBeakSettings()
        config = settings.model_config

        assert "env_prefix" in config
        assert config["env_prefix"] == "DATABEAK_"
        assert config.get("case_sensitive", True) is False

    def test_environment_variables_mapping(self):
        """Test that documented environment variables map to settings fields."""
        settings = DataBeakSettings()

        # Verify all documented environment variables have corresponding fields
        documented_vars = {
            "DATABEAK_MAX_FILE_SIZE_MB": "max_file_size_mb",
            "DATABEAK_CSV_HISTORY_DIR": "csv_history_dir",
            "DATABEAK_SESSION_TIMEOUT": "session_timeout",
            "DATABEAK_CHUNK_SIZE": "chunk_size",
            "DATABEAK_AUTO_SAVE": "auto_save",
        }

        for env_var, field_name in documented_vars.items():
            # Check that the field exists in the settings model
            assert hasattr(settings, field_name), f"Field {field_name} missing for {env_var}"

    def test_settings_default_values(self):
        """Test that settings have sensible defaults."""
        settings = DataBeakSettings()

        assert settings.max_file_size_mb == 1024
        assert settings.csv_history_dir == "."
        assert settings.session_timeout == 3600
        assert settings.chunk_size == 10000
        assert settings.auto_save is True

    def test_environment_variable_override(self, monkeypatch):
        """Test that environment variables properly override defaults."""
        # Set test environment variables
        monkeypatch.setenv("DATABEAK_MAX_FILE_SIZE_MB", "2048")
        monkeypatch.setenv("DATABEAK_CSV_HISTORY_DIR", "/tmp/test")
        monkeypatch.setenv("DATABEAK_SESSION_TIMEOUT", "7200")
        monkeypatch.setenv("DATABEAK_CHUNK_SIZE", "5000")
        monkeypatch.setenv("DATABEAK_AUTO_SAVE", "false")

        # Create new settings instance to pick up env vars
        settings = DataBeakSettings()

        assert settings.max_file_size_mb == 2048
        assert settings.csv_history_dir == "/tmp/test"
        assert settings.session_timeout == 7200
        assert settings.chunk_size == 5000
        assert settings.auto_save is False


class TestCoverageConfiguration:
    """Test coverage configuration is valid."""

    def test_coverage_omit_paths_exist_or_are_valid_patterns(self):
        """Test that coverage omit paths are valid."""
        import tomllib

        with Path("pyproject.toml").open("rb") as f:
            config = tomllib.load(f)

        omit_patterns = config["tool"]["coverage"]["run"]["omit"]

        # All omit patterns should be valid glob patterns or existing paths
        for pattern in omit_patterns:
            if pattern.startswith("*/"):
                # Glob pattern - should be valid
                assert "*" in pattern
            else:
                # Specific path - should exist or be removed
                path = Path(pattern)
                # For this test, we allow non-existent paths if they're clearly patterns
                if not path.exists():
                    assert "*" in pattern or pattern.startswith(
                        "src/"
                    ), f"Non-existent specific path in coverage omit: {pattern}"

    def test_coverage_source_paths_exist(self):
        """Test that coverage source paths exist."""
        import tomllib

        with Path("pyproject.toml").open("rb") as f:
            config = tomllib.load(f)

        source_paths = config["tool"]["coverage"]["run"]["source"]

        for source_path in source_paths:
            path = Path(source_path)
            assert path.exists(), f"Coverage source path doesn't exist: {source_path}"
            assert path.is_dir(), f"Coverage source path is not a directory: {source_path}"


class TestProjectStructureConsistency:
    """Test that project structure matches documentation."""

    def test_documented_directories_exist(self):
        """Test that directories mentioned in README project structure exist."""
        documented_dirs = [
            "src/databeak",
            "src/databeak/models",
            "src/databeak/tools",
            "src/databeak/resources",
            "src/databeak/utils",
            "src/databeak/prompts",
            "tests",
            "examples",
            "scripts",
            "docs",
        ]

        for dir_path in documented_dirs:
            path = Path(dir_path)
            assert path.exists(), f"Documented directory doesn't exist: {dir_path}"
            assert path.is_dir(), f"Documented path is not a directory: {dir_path}"

    def test_documented_key_files_exist(self):
        """Test that key files mentioned in project structure exist."""
        documented_files = [
            "src/databeak/server.py",
            "src/databeak/models/csv_session.py",
            "src/databeak/models/data_models.py",
            "src/databeak/models/data_session.py",
            "src/databeak/exceptions.py",
            "src/databeak/_version.py",
        ]

        for file_path in documented_files:
            path = Path(file_path)
            assert path.exists(), f"Documented file doesn't exist: {file_path}"
            assert path.is_file(), f"Documented path is not a file: {file_path}"
