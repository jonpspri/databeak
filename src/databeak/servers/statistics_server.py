"""Standalone Statistics server for DataBeak using FastMCP server composition.

This module provides a complete Statistics server implementation following DataBeak's modular server
architecture pattern. It focuses on core statistical analysis, numerical computations, and
correlation analysis with optimized mathematical processing.
"""

from __future__ import annotations

import logging
from typing import Literal, cast

import numpy as np
import pandas as pd
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

# Import session management and data models from the main package
from ..models import OperationType, get_session_manager
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
    ctx: Context | None = None,  # noqa: ARG001
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
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session or not session.data_session.has_data():
            raise ToolError(f"Invalid session or no data loaded: {session_id}")

        df = session.data_session.df
        assert df is not None  # Type guard: has_data() ensures df is not None

        # Select columns to analyze
        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ToolError(f"Columns not found: {missing_cols}")
            numeric_df = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            # Return basic statistics for empty numeric data
            return StatisticsResult(
                session_id=session_id,
                statistics={},
                column_count=0,
                numeric_columns=[],
                total_rows=len(df),
            )

        # Calculate statistics
        stats = {}

        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()

            # Build the StatisticsSummary object
            percentile_25 = float(col_data.quantile(0.25)) if include_percentiles else 0.0
            percentile_50 = float(col_data.quantile(0.50)) if include_percentiles else 0.0
            percentile_75 = float(col_data.quantile(0.75)) if include_percentiles else 0.0

            col_stats = StatisticsSummary(
                count=int(col_data.count()),
                mean=float(col_data.mean()),
                std=float(col_data.std()),
                min=float(col_data.min()),
                **{"25%": percentile_25, "50%": percentile_50, "75%": percentile_75},
                max=float(col_data.max()),
            )

            stats[col] = col_stats

        session.record_operation(
            OperationType.ANALYZE, {"type": "statistics", "columns": list(stats.keys())}
        )

        return StatisticsResult(
            session_id=session_id,
            statistics=stats,
            column_count=len(stats),
            numeric_columns=list(stats.keys()),
            total_rows=len(df),
        )

    except Exception as e:
        logger.error(f"Error getting statistics: {e!s}")
        raise ToolError(f"Error getting statistics: {e}") from e


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
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session or not session.data_session.has_data():
            raise ToolError(f"Invalid session or no data loaded: {session_id}")

        df = session.data_session.df
        assert df is not None  # Type guard: has_data() ensures df is not None

        if column not in df.columns:
            raise ToolError(f"Column '{column}' not found")

        col_data = df[column]
        # Map pandas dtypes to Pydantic model literals using type-safe mapping
        col_dtype = str(col_data.dtype)

        # Direct mapping for exact matches, with type-safe fallback for partial matches
        dtype_mapping: dict[
            str, Literal["int64", "float64", "object", "bool", "datetime64", "category"]
        ] = {
            "int64": "int64",
            "float64": "float64",
            "bool": "bool",
            "object": "object",
            "category": "category",
        }

        # Handle exact matches first
        if col_dtype in dtype_mapping:
            mapped_dtype = dtype_mapping[col_dtype]
        elif col_dtype.startswith("datetime64"):  # Handle datetime64[ns] variants
            mapped_dtype = "datetime64"
        else:
            mapped_dtype = "object"  # Safe fallback

        # Create statistics - only meaningful for numeric columns (excluding boolean)
        if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
            non_null = col_data.dropna()
            if len(non_null) > 0:
                statistics = StatisticsSummary(
                    count=int(non_null.count()),
                    mean=float(non_null.mean()),
                    std=float(non_null.std()),
                    min=float(non_null.min()),
                    **{
                        "25%": float(non_null.quantile(0.25)),
                        "50%": float(non_null.quantile(0.50)),
                        "75%": float(non_null.quantile(0.75)),
                    },
                    max=float(non_null.max()),
                )
            else:
                # Empty numeric column
                statistics = StatisticsSummary(
                    count=0,
                    mean=0.0,
                    std=0.0,
                    min=0.0,
                    **{"25%": 0.0, "50%": 0.0, "75%": 0.0},
                    max=0.0,
                )
        else:
            # For non-numeric columns, create placeholder statistics
            statistics = StatisticsSummary(
                count=int(col_data.notna().sum()),
                mean=0.0,
                std=0.0,
                min=0.0,
                **{"25%": 0.0, "50%": 0.0, "75%": 0.0},
                max=0.0,
            )

        session.record_operation(
            OperationType.ANALYZE, {"type": "column_statistics", "column": column}
        )

        return ColumnStatisticsResult(
            session_id=session_id,
            column=column,
            statistics=statistics,
            data_type=mapped_dtype,
            non_null_count=int(col_data.notna().sum()),
        )

    except Exception as e:
        logger.error(f"Column statistics failed: {e}")
        if ctx:
            await ctx.error(f"Column statistics failed: {e!s}")
        raise ToolError(f"Failed to analyze column '{column}': {e}") from e


async def get_correlation_matrix(
    session_id: str,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    columns: list[str] | None = None,
    min_correlation: float | None = None,
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
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session or not session.data_session.has_data():
            raise ToolError(f"Invalid session or no data loaded: {session_id}")

        df = session.data_session.df
        assert df is not None  # Type guard: has_data() ensures df is not None

        # Select columns
        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ToolError(f"Columns not found: {missing_cols}")
            numeric_df = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ToolError("No numeric columns found")

        if len(numeric_df.columns) < 2:
            raise ToolError("Need at least 2 numeric columns for correlation")

        # Calculate correlation
        if method not in ["pearson", "spearman", "kendall"]:
            raise ToolError(f"Invalid method: {method}")

        corr_matrix = numeric_df.corr(method=method)

        # Convert to dict format
        correlations: dict[str, dict[str, float]] = {}
        for col1 in corr_matrix.columns:
            correlations[col1] = {}
            for col2 in corr_matrix.columns:
                value = corr_matrix.loc[col1, col2]
                if not pd.isna(value):
                    float_value = float(cast("float", value))
                    if (
                        min_correlation is None
                        or abs(float_value) >= min_correlation
                        or col1 == col2
                    ):
                        correlations[col1][col2] = round(float_value, 4)

        session.record_operation(
            OperationType.ANALYZE,
            {
                "type": "correlation",
                "method": method,
                "columns": list(corr_matrix.columns),
            },
        )

        return CorrelationResult(
            session_id=session_id,
            correlation_matrix=correlations,
            method=method,
            columns_analyzed=list(corr_matrix.columns),
        )

    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        if ctx:
            await ctx.error(f"Correlation analysis failed: {e!s}")
        raise ToolError(f"Failed to compute correlation matrix: {e}") from e


async def get_value_counts(
    session_id: str,
    column: str,
    normalize: bool = False,
    sort: bool = True,
    ascending: bool = False,
    top_n: int | None = None,
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
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session or not session.data_session.has_data():
            raise ToolError(f"Invalid session or no data loaded: {session_id}")

        df = session.data_session.df
        assert df is not None  # Type guard: has_data() ensures df is not None

        if column not in df.columns:
            raise ToolError(f"Column '{column}' not found")

        # Get value counts
        value_counts: pd.Series[int] | pd.Series[float]
        if normalize:
            value_counts = df[column].value_counts(
                normalize=True, sort=sort, ascending=ascending, dropna=False
            )
        else:
            value_counts = df[column].value_counts(
                normalize=False, sort=sort, ascending=ascending, dropna=False
            )

        # Apply top_n if specified
        if top_n:
            value_counts = value_counts.head(top_n)

        # Convert to dict
        counts_dict = {}
        for value, count in value_counts.items():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                key = "NaN"
            else:
                key = str(value)
            counts_dict[key] = float(count) if normalize else int(count)

        # Calculate additional statistics
        unique_count = df[column].nunique(dropna=False)

        session.record_operation(
            OperationType.ANALYZE,
            {
                "type": "value_counts",
                "column": column,
                "normalize": normalize,
                "top_n": top_n,
            },
        )

        return ValueCountsResult(
            session_id=session_id,
            column=column,
            value_counts=counts_dict,
            total_values=len(df),
            unique_values=int(unique_count),
            normalize=normalize,
        )

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
