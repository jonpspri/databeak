"""Statistics-specific Pydantic models for DataBeak.

This module contains response models for statistical operations, separated to avoid circular imports
between servers and services.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .tool_responses import BaseToolResponse


class StatisticsSummary(BaseModel):
    """Statistical summary for a single column."""

    model_config = ConfigDict(populate_by_name=True)

    count: int
    mean: float | None = None
    std: float | None = None
    min: float | str | None = None
    percentile_25: float | None = Field(default=None, alias="25%")
    percentile_50: float | None = Field(default=None, alias="50%")
    percentile_75: float | None = Field(default=None, alias="75%")
    max: float | str | None = None

    # Categorical statistics fields
    unique: int | None = None
    top: str | None = None
    freq: int | None = None


class StatisticsResult(BaseToolResponse):
    """Response model for dataset statistical analysis."""

    session_id: str
    statistics: dict[str, StatisticsSummary]
    column_count: int
    numeric_columns: list[str]
    total_rows: int


class ColumnStatisticsResult(BaseToolResponse):
    """Response model for individual column statistical analysis."""

    session_id: str
    column: str
    statistics: StatisticsSummary
    data_type: Literal["int64", "float64", "object", "bool", "datetime64", "category"]
    non_null_count: int


class CorrelationResult(BaseToolResponse):
    """Response model for correlation matrix analysis."""

    session_id: str
    correlation_matrix: dict[str, dict[str, float]]
    method: Literal["pearson", "spearman", "kendall"]
    columns_analyzed: list[str]


class ValueCountsResult(BaseToolResponse):
    """Response model for value frequency analysis."""

    session_id: str
    column: str
    value_counts: dict[str, int | float]
    total_values: int
    unique_values: int
    normalize: bool = False
