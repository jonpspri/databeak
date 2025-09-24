"""Configuration settings for DataBeak."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class DataBeakSettings(BaseSettings):
    """Configuration settings for session management."""

    max_file_size_mb: int = Field(default=1024, description="Maximum file size limit in megabytes")
    session_timeout: int = Field(default=3600, description="Session timeout in seconds")
    chunk_size: int = Field(
        default=10000,
        description="Default chunk size for processing large datasets",
    )
    memory_threshold_mb: int = Field(
        default=2048, description="Memory usage threshold in MB for health monitoring"
    )
    memory_warning_threshold: float = Field(
        default=0.75, description="Memory usage ratio that triggers warning status (0.0-1.0)"
    )
    memory_critical_threshold: float = Field(
        default=0.90, description="Memory usage ratio that triggers critical status (0.0-1.0)"
    )
    session_capacity_warning_threshold: float = Field(
        default=0.90, description="Session capacity ratio that triggers warning (0.0-1.0)"
    )
    max_validation_violations: int = Field(
        default=1000, description="Maximum number of validation violations to report"
    )
    max_anomaly_sample_size: int = Field(
        default=10000, description="Maximum sample size for anomaly detection operations"
    )

    # Encoding detection thresholds
    encoding_confidence_threshold: float = Field(
        default=0.7, description="Minimum confidence threshold for encoding detection"
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


_settings: DataBeakSettings | None = None


def create_settings() -> DataBeakSettings:
    """Create a new DataBeak settings instance."""
    return DataBeakSettings()


def get_settings() -> DataBeakSettings:
    """Create or get the global DataBeak settings instance."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = create_settings()
    return _settings


def reset_settings() -> None:
    """Reset the global DataBeak settings instance."""
    global _settings  # noqa: PLW0603
    _settings = None
