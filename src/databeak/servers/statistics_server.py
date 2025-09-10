"""Standalone Statistics server for DataBeak using FastMCP server composition.

This module provides a complete Statistics server implementation following DataBeak's modular server
architecture pattern. It focuses on core statistical analysis, numerical computations, and
correlation analysis with optimized mathematical processing.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

# Import session management and data models from the main package
from ..models.tool_responses import BaseToolResponse

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS FOR STATISTICS OPERATIONS
# ============================================================================


class StatisticsSummary(BaseModel):
    """Statistical summary for a single column."""

    model_config = ConfigDict(populate_by_name=True)

    count: int
    mean: float
    std: float
    min: float
    percentile_25: float = Field(alias="25%")
    percentile_50: float = Field(alias="50%")
    percentile_75: float = Field(alias="75%")
    max: float


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
    value_counts: dict[str, int]
    total_values: int
    unique_values: int
    normalize: bool = False


# ============================================================================
# STATISTICAL OPERATIONS LOGIC
# ============================================================================


async def get_statistics(
    session_id: str,
    columns: list[str] | None = None,
    include_percentiles: bool = True,
    ctx: Context | None = None,
) -> StatisticsResult:
    """Get comprehensive statistical summary of numerical columns.

    Computes descriptive statistics for all or specified numerical columns including
    count, mean, standard deviation, min/max values, and percentiles. Optimized for
    AI workflows with clear statistical insights and data understanding.

    Args:
        session_id: Session ID containing loaded data
        columns: Optional list of specific columns to analyze (default: all numeric)
        include_percentiles: Whether to include 25th, 50th, 75th percentiles
        ctx: FastMCP context for progress reporting

    Returns:
        Comprehensive statistical analysis with per-column summaries

    Statistical Metrics:
        📊 Count: Number of non-null values
        📈 Mean: Average value
        📉 Std: Standard deviation (measure of spread)
        🔢 Min/Max: Minimum and maximum values
        📊 Percentiles: 25th, 50th (median), 75th quartiles

    Examples:
        # Get statistics for all numeric columns
        stats = await get_statistics("session_123")

        # Analyze specific columns only
        stats = await get_statistics("session_123", columns=["price", "quantity"])

        # Skip percentiles for faster computation
        stats = await get_statistics("session_123", include_percentiles=False)

    AI Workflow Integration:
        1. Essential for data understanding and quality assessment
        2. Identifies data distribution and potential issues
        3. Guides feature engineering and analysis decisions
        4. Provides context for outlier detection thresholds
    """
    try:
        # Import core analytics function
        from ..tools.analytics import get_statistics as _get_statistics

        return await _get_statistics(session_id, columns, include_percentiles, ctx)

    except Exception as e:
        logger.error(f"Statistics analysis failed: {e}")
        if ctx:
            await ctx.error(f"Statistics analysis failed: {e!s}")
        raise ToolError(f"Failed to compute statistics: {e}") from e


async def get_column_statistics(
    session_id: str,
    column: str,
    ctx: Context | None = None,
) -> ColumnStatisticsResult:
    """Get detailed statistical analysis for a single column.

    Provides focused statistical analysis for a specific column including
    data type information, null value handling, and comprehensive numerical
    statistics when applicable.

    Args:
        session_id: Session ID containing loaded data
        column: Name of the column to analyze
        ctx: FastMCP context for progress reporting

    Returns:
        Detailed statistical analysis for the specified column

    Column Analysis:
        🔍 Data Type: Detected pandas data type
        📊 Statistics: Complete statistical summary for numeric columns
        🔢 Non-null Count: Number of valid (non-null) values
        📈 Distribution: Statistical distribution characteristics

    Examples:
        # Analyze a price column
        stats = await get_column_statistics("session_123", "price")

        # Analyze a categorical column
        stats = await get_column_statistics("session_123", "category")

    AI Workflow Integration:
        1. Deep dive analysis for specific columns of interest
        2. Data quality assessment for individual features
        3. Understanding column characteristics for modeling
        4. Validation of data transformations
    """
    try:
        from ..tools.analytics import get_column_statistics as _get_column_statistics

        return await _get_column_statistics(session_id, column, ctx)

    except Exception as e:
        logger.error(f"Column statistics failed: {e}")
        if ctx:
            await ctx.error(f"Column statistics failed: {e!s}")
        raise ToolError(f"Failed to analyze column '{column}': {e}") from e


async def get_correlation_matrix(
    session_id: str,
    columns: list[str] | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    ctx: Context | None = None,
) -> CorrelationResult:
    """Calculate correlation matrix for numerical columns.

    Computes pairwise correlations between numerical columns using various
    correlation methods. Essential for understanding relationships between
    variables and feature selection in analytical workflows.

    Args:
        session_id: Session ID containing loaded data
        columns: Optional list of columns to include (default: all numeric)
        method: Correlation method - pearson (linear), spearman (rank), kendall (rank)
        ctx: FastMCP context for progress reporting

    Returns:
        Correlation matrix with pairwise correlation coefficients

    Correlation Methods:
        📊 Pearson: Linear relationships (default, assumes normality)
        📈 Spearman: Monotonic relationships (rank-based, non-parametric)
        🔄 Kendall: Concordant/discordant pairs (robust, small samples)

    Examples:
        # Basic correlation analysis
        corr = await get_correlation_matrix("session_123")

        # Analyze specific columns with Spearman correlation
        corr = await get_correlation_matrix("session_123",
                                          columns=["price", "rating", "sales"],
                                          method="spearman")

    AI Workflow Integration:
        1. Feature selection and dimensionality reduction
        2. Multicollinearity detection before modeling
        3. Understanding variable relationships
        4. Data validation and quality assessment
    """
    try:
        from ..tools.analytics import get_correlation_matrix as _get_correlation_matrix

        return await _get_correlation_matrix(session_id, columns, method, ctx)

    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        if ctx:
            await ctx.error(f"Correlation analysis failed: {e!s}")
        raise ToolError(f"Failed to compute correlation matrix: {e}") from e


async def get_value_counts(
    session_id: str,
    column: str,
    normalize: bool = False,
    sort_desc: bool = True,
    limit: int | None = None,
    ctx: Context | None = None,
) -> ValueCountsResult:
    """Get frequency distribution of values in a column.

    Analyzes the distribution of values in a specified column, providing
    counts and optionally percentages for each unique value. Essential for
    understanding categorical data and identifying common patterns.

    Args:
        session_id: Session ID containing loaded data
        column: Name of the column to analyze
        normalize: If True, return percentages instead of counts
        sort_desc: Sort results by frequency (descending if True)
        limit: Maximum number of values to return (default: all)
        ctx: FastMCP context for progress reporting

    Returns:
        Frequency distribution with counts/percentages for each unique value

    Analysis Features:
        🔢 Frequency Counts: Raw counts for each unique value
        📊 Percentage Mode: Normalized frequencies as percentages
        🎯 Top Values: Configurable limit for most frequent values
        📈 Summary Stats: Total values, unique count, distribution insights

    Examples:
        # Basic value counts
        counts = await get_value_counts("session_123", "category")

        # Get percentages for top 10 values
        counts = await get_value_counts("session_123", "status",
                                      normalize=True, limit=10)

    AI Workflow Integration:
        1. Categorical data analysis and encoding decisions
        2. Data quality assessment (identifying rare values)
        3. Understanding distribution for sampling strategies
        4. Feature engineering insights for categorical variables
    """
    try:
        from ..tools.analytics import get_value_counts as _get_value_counts

        return await _get_value_counts(session_id, column, normalize, sort_desc, limit, ctx)

    except Exception as e:
        logger.error(f"Value counts analysis failed: {e}")
        if ctx:
            await ctx.error(f"Value counts analysis failed: {e!s}")
        raise ToolError(f"Failed to analyze value counts for column '{column}': {e}") from e


# ============================================================================
# FASTMCP SERVER SETUP
# ============================================================================


# Create Statistics server
statistics_server = FastMCP(
    "DataBeak-Statistics",
    instructions="Statistics and correlation analysis server for DataBeak with comprehensive numerical analysis capabilities",
)


# Register the statistical analysis functions directly as MCP tools
statistics_server.tool(name="get_statistics")(get_statistics)
statistics_server.tool(name="get_column_statistics")(get_column_statistics)
statistics_server.tool(name="get_correlation_matrix")(get_correlation_matrix)
statistics_server.tool(name="get_value_counts")(get_value_counts)
