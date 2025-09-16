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

    count: int = Field(description="Total number of non-null values")
    mean: float | None = Field(default=None, description="Arithmetic mean (numeric columns only)")
    std: float | None = Field(default=None, description="Standard deviation (numeric columns only)")
    min: float | str | None = Field(default=None, description="Minimum value in the column")
    percentile_25: float | None = Field(
        default=None, alias="25%", description="25th percentile value (numeric columns only)"
    )
    percentile_50: float | None = Field(
        default=None, alias="50%", description="50th percentile/median value (numeric columns only)"
    )
    percentile_75: float | None = Field(
        default=None, alias="75%", description="75th percentile value (numeric columns only)"
    )
    max: float | str | None = Field(default=None, description="Maximum value in the column")

    # Categorical statistics fields
    unique: int | None = Field(
        None, description="Number of unique values (categorical columns only)"
    )
    top: str | None = Field(
        None, description="Most frequently occurring value (categorical columns only)"
    )
    freq: int | None = Field(
        None, description="Frequency of the most common value (categorical columns only)"
    )


class StatisticsResult(BaseToolResponse):
    """Response model for dataset statistical analysis."""
    statistics: dict[str, StatisticsSummary] = Field(
        description="Statistical summary for each column"
    )
    column_count: int = Field(description="Total number of columns analyzed")
    numeric_columns: list[str] = Field(description="Names of numeric columns that were analyzed")
    total_rows: int = Field(description="Total number of rows in the dataset")


class ColumnStatisticsResult(BaseToolResponse):
    """Response model for individual column statistical analysis."""
    column: str = Field(description="Name of the analyzed column")
    statistics: StatisticsSummary = Field(description="Statistical summary for the column")
    data_type: Literal["int64", "float64", "object", "bool", "datetime64", "category"] = Field(
        description="Pandas data type of the column"
    )
    non_null_count: int = Field(description="Number of non-null values in the column")


class CorrelationResult(BaseToolResponse):
    """Response model for correlation matrix analysis."""
    correlation_matrix: dict[str, dict[str, float]] = Field(
        description="Correlation coefficients between columns"
    )
    method: Literal["pearson", "spearman", "kendall"] = Field(
        description="Correlation method used for analysis"
    )
    columns_analyzed: list[str] = Field(
        description="Names of columns included in correlation analysis"
    )


class ValueCountsResult(BaseToolResponse):
    """Response model for value frequency analysis."""
    column: str = Field(description="Name of the analyzed column")
    value_counts: dict[str, int | float] = Field(
        description="Count or proportion of each unique value"
    )
    total_values: int = Field(description="Total number of values (including duplicates)")
    unique_values: int = Field(description="Number of unique/distinct values")
    normalize: bool = Field(
        default=False, description="Whether counts are normalized as proportions"
    )
