"""Configuration settings for DataBeak."""

from __future__ import annotations

import threading

from pydantic import Field
from pydantic_settings import BaseSettings


class DatabeakSettings(BaseSettings):
    """Configuration settings for DataBeak operations.

    Settings are organized into categories:

    Session Management:
    - session_timeout: How long sessions stay alive
    - session_capacity_warning_threshold: When to warn about session capacity

    Health Monitoring (for health_check tool):
    - health_memory_threshold_mb: Total server memory limit for health status
    - memory_warning_threshold: Ratio that triggers "degraded" status (75%)
    - memory_critical_threshold: Ratio that triggers "unhealthy" status (90%)

    Data Loading Limits (enforced during load_csv_from_url/load_csv_from_content):
    - max_memory_usage_mb: Hard limit for individual DataFrame memory
    - max_rows: Hard limit for DataFrame row count
    - url_timeout_seconds: Network timeout for URL downloads
    - max_download_size_mb: Maximum download size from URLs

    Data Validation:
    - Various thresholds for quality checks and anomaly detection
    """

    # Session management
    session_timeout: int = Field(default=3600, description="Session timeout in seconds")
    session_capacity_warning_threshold: float = Field(
        default=0.90, description="Session capacity ratio that triggers warning (0.0-1.0)"
    )

    # Health monitoring thresholds (used by health_check for server status)
    health_memory_threshold_mb: int = Field(
        default=2048,
        description="Total server memory threshold in MB for health status monitoring (not a hard limit)",
    )
    memory_warning_threshold: float = Field(
        default=0.75,
        description="Memory usage ratio that triggers 'degraded' health status (0.0-1.0)",
    )
    memory_critical_threshold: float = Field(
        default=0.90,
        description="Memory usage ratio that triggers 'unhealthy' health status (0.0-1.0)",
    )

    # Data loading limits (hard limits enforced during CSV loading operations)
    max_memory_usage_mb: int = Field(
        default=1000,
        description="Maximum memory in MB for individual DataFrames (hard limit, loading fails if exceeded)",
    )
    max_rows: int = Field(
        default=1_000_000,
        description="Maximum rows per DataFrame (hard limit, loading fails if exceeded)",
    )
    url_timeout_seconds: int = Field(
        default=30, description="Network timeout for URL downloads in seconds"
    )
    max_download_size_mb: int = Field(
        default=100, description="Maximum download size for URLs in MB (hard limit)"
    )

    # Data validation and analysis
    max_validation_violations: int = Field(
        default=1000, description="Maximum number of validation violations to report"
    )
    max_anomaly_sample_size: int = Field(
        default=10000, description="Maximum sample size for anomaly detection operations"
    )

    # Data validation thresholds
    data_completeness_threshold: float = Field(
        default=0.5, description="Threshold for determining if data is complete enough"
    )
    outlier_detection_threshold: float = Field(
        default=0.1, description="Threshold for outlier detection in data validation"
    )
    uniqueness_threshold: float = Field(
        default=0.99, description="Threshold for determining if values are sufficiently unique"
    )
    high_quality_threshold: float = Field(
        default=0.9, description="Threshold for determining high quality data"
    )
    correlation_threshold: float = Field(
        default=0.3, description="Threshold for correlation analysis"
    )

    # Count-based thresholds
    min_statistical_sample_size: int = Field(
        default=2, description="Minimum sample size for statistical operations"
    )
    character_score_threshold: int = Field(
        default=85, description="Character encoding quality score threshold"
    )
    max_category_display: int = Field(
        default=10, description="Maximum number of categories to display in summaries"
    )
    min_length_threshold: int = Field(
        default=7, description="Minimum length threshold for data validation"
    )
    percentage_multiplier: int = Field(
        default=100, description="Multiplier for converting ratios to percentages"
    )

    model_config = {"env_prefix": "DATABEAK_", "case_sensitive": False}


_settings: DatabeakSettings | None = None
_lock = threading.Lock()


def create_settings() -> DatabeakSettings:
    """Create a new DataBeak settings instance."""
    return DatabeakSettings()


def get_settings() -> DatabeakSettings:
    """Create or get the global DataBeak settings instance."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        with _lock:
            if _settings is None:
                _settings = create_settings()
    return _settings


def reset_settings() -> None:
    """Reset the global DataBeak settings instance."""
    global _settings  # noqa: PLW0603
    with _lock:
        _settings = None
