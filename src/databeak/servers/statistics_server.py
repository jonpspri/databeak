"""Standalone Statistics server for DataBeak using FastMCP server composition.

This module provides a complete Statistics server implementation following DataBeak's modular server
architecture pattern. It focuses on core statistical analysis, numerical computations, and
correlation analysis with optimized mathematical processing.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

# Import session management and data models from the main package
from ..exceptions import (
    ColumnNotFoundError,
    InvalidParameterError,
    NoDataLoadedError,
    SessionNotFoundError,
)
from ..models import OperationType, get_session_manager

# Import response models - needed at runtime for FastMCP
from ..models.statistics_models import (
    ColumnStatisticsResult,
    CorrelationResult,
    StatisticsResult,
    ValueCountsResult,
)

logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _get_session_data(session_id: str) -> tuple[Any, pd.DataFrame]:
    """Get session and DataFrame, raising appropriate exceptions if not found."""
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if not session:
        raise SessionNotFoundError(session_id)
    if not session.data_session.has_data():
        raise NoDataLoadedError(session_id)

    df = session.data_session.df
    if df is None:  # Type guard since has_data() was checked
        raise NoDataLoadedError(session_id)
    return session, df


# ============================================================================
# STATISTICAL OPERATIONS LOGIC - DIRECT IMPLEMENTATIONS
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
        session, df = _get_session_data(session_id)

        # Select numeric columns
        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ColumnNotFoundError(missing_cols[0], df.columns.tolist())
            numeric_df = df[columns].select_dtypes(include=[np.number])
            if numeric_df.empty:
                raise InvalidParameterError(
                    "columns", str(columns), "No numeric columns found in specified columns"
                )
        else:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                raise InvalidParameterError(
                    "data", "dataframe", "No numeric columns found in the dataset"
                )

        # Calculate statistics
        stats_dict = {}
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()

            # Create StatisticsSummary directly
            from ..models.statistics_models import StatisticsSummary

            # Calculate statistics, using 0.0 for undefined values
            col_stats = StatisticsSummary(
                count=int(col_data.count()),
                mean=float(col_data.mean())
                if len(col_data) > 0 and not pd.isna(col_data.mean())
                else 0.0,
                std=float(col_data.std())
                if len(col_data) > 1 and not pd.isna(col_data.std())
                else 0.0,
                min=float(col_data.min())
                if len(col_data) > 0 and not pd.isna(col_data.min())
                else 0.0,
                max=float(col_data.max())
                if len(col_data) > 0 and not pd.isna(col_data.max())
                else 0.0,
                **{
                    "25%": float(col_data.quantile(0.25)) if len(col_data) > 0 else 0.0,
                    "50%": float(col_data.quantile(0.50)) if len(col_data) > 0 else 0.0,
                    "75%": float(col_data.quantile(0.75)) if len(col_data) > 0 else 0.0,
                },
            )

            stats_dict[col] = col_stats

        session.record_operation(
            OperationType.ANALYZE,
            {
                "type": "statistics",
                "columns": list(stats_dict.keys()),
                "include_percentiles": include_percentiles,
            },
        )

        return StatisticsResult(
            session_id=session_id,
            statistics=stats_dict,
            column_count=len(stats_dict),
            numeric_columns=list(stats_dict.keys()),
            total_rows=len(df),
        )

    except (
        SessionNotFoundError,
        NoDataLoadedError,
        ColumnNotFoundError,
        InvalidParameterError,
    ) as e:
        logger.error(f"Statistics calculation failed: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Error calculating statistics: {e!s}")
        raise ToolError(f"Error calculating statistics: {e}") from e


async def get_column_statistics(
    session_id: str,
    column: str,
    ctx: Context | None = None,  # noqa: ARG001
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
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ColumnNotFoundError(column, df.columns.tolist())

        col_data = df[column]
        dtype = str(col_data.dtype)
        count = int(col_data.count())
        null_count = int(col_data.isnull().sum())
        unique_count = int(col_data.nunique())

        # Initialize statistics dict (with mixed types for flexibility)
        statistics: dict[str, Any] = {
            "count": count,
            "null_count": null_count,
            "unique_count": unique_count,
        }

        # Add numeric statistics if column is numeric
        if pd.api.types.is_numeric_dtype(col_data):
            statistics.update(
                {
                    "mean": float(col_data.mean()) if not pd.isna(col_data.mean()) else 0.0,
                    "std": float(col_data.std()) if not pd.isna(col_data.std()) else 0.0,
                    "min": float(col_data.min()) if not pd.isna(col_data.min()) else 0.0,
                    "max": float(col_data.max()) if not pd.isna(col_data.max()) else 0.0,
                    "25%": float(col_data.quantile(0.25))
                    if not pd.isna(col_data.quantile(0.25))
                    else 0.0,
                    "50%": float(col_data.quantile(0.50))
                    if not pd.isna(col_data.quantile(0.50))
                    else 0.0,
                    "75%": float(col_data.quantile(0.75))
                    if not pd.isna(col_data.quantile(0.75))
                    else 0.0,
                }
            )
        else:
            # For non-numeric columns, add most frequent value
            if count > 0:
                mode_result = col_data.mode()
                most_frequent = mode_result.iloc[0] if len(mode_result) > 0 else None
                if most_frequent is not None and not pd.isna(most_frequent):
                    statistics["most_frequent"] = str(most_frequent)
                    statistics["most_frequent_count"] = int(col_data.value_counts().iloc[0])

        session.record_operation(
            OperationType.ANALYZE,
            {
                "type": "column_statistics",
                "column": column,
                "dtype": dtype,
            },
        )

        # Convert statistics dict to StatisticsSummary if numeric
        if pd.api.types.is_numeric_dtype(col_data):
            from ..models.statistics_models import StatisticsSummary

            stats_summary = StatisticsSummary(
                count=statistics["count"],
                mean=statistics.get("mean", 0.0),
                std=statistics.get("std", 0.0),
                min=statistics.get("min", 0.0),
                **{
                    "25%": statistics.get("25%", 0.0),
                    "50%": statistics.get("50%", 0.0),
                    "75%": statistics.get("75%", 0.0),
                },
                max=statistics.get("max", 0.0),
            )
        else:
            # For non-numeric columns, create a basic StatisticsSummary
            from ..models.statistics_models import StatisticsSummary

            stats_summary = StatisticsSummary(
                count=statistics["count"],
                mean=0.0,
                std=0.0,
                min=0.0,
                **{"25%": 0.0, "50%": 0.0, "75%": 0.0},
                max=0.0,
            )

        # Map dtype to expected literal type
        dtype_map: dict[
            str, Literal["int64", "float64", "object", "bool", "datetime64", "category"]
        ] = {
            "int64": "int64",
            "float64": "float64",
            "object": "object",
            "bool": "bool",
            "datetime64[ns]": "datetime64",
            "category": "category",
        }
        data_type: Literal["int64", "float64", "object", "bool", "datetime64", "category"] = (
            dtype_map.get(dtype, "object")
        )

        return ColumnStatisticsResult(
            session_id=session_id,
            column=column,
            statistics=stats_summary,
            data_type=data_type,
            non_null_count=count,
        )

    except (SessionNotFoundError, NoDataLoadedError, ColumnNotFoundError) as e:
        logger.error(f"Column statistics failed: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Error calculating column statistics: {e!s}")
        raise ToolError(f"Error calculating column statistics: {e}") from e


async def get_correlation_matrix(
    session_id: str,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    columns: list[str] | None = None,
    min_correlation: float | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> CorrelationResult:
    """Calculate correlation matrix for numerical columns.

    Computes pairwise correlations between numerical columns using various
    correlation methods. Essential for understanding relationships between
    variables and feature selection in analytical workflows.

    Args:
        session_id: Session ID containing loaded data
        columns: Optional list of columns to include (default: all numeric)
        method: Correlation method - pearson (linear), spearman (rank), kendall (rank)
        min_correlation: Minimum correlation threshold to include in results
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

        # Filter correlations above threshold
        corr = await get_correlation_matrix("session_123", min_correlation=0.5)

    AI Workflow Integration:
        1. Feature selection and dimensionality reduction
        2. Multicollinearity detection before modeling
        3. Understanding variable relationships
        4. Data validation and quality assessment
    """
    try:
        session, df = _get_session_data(session_id)

        # Select numeric columns
        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ColumnNotFoundError(missing_cols[0], df.columns.tolist())
            numeric_df = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise InvalidParameterError(
                "data", "dataframe", "No numeric columns found for correlation analysis"
            )

        if len(numeric_df.columns) < 2:
            raise InvalidParameterError(
                "columns",
                str(list(numeric_df.columns)),
                "At least 2 numeric columns required for correlation",
            )

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=method)

        # Convert to dict format
        correlation_dict: dict[str, dict[str, float]] = {}
        for col1 in corr_matrix.columns:
            correlation_dict[col1] = {}
            for col2 in corr_matrix.columns:
                corr_val = corr_matrix.loc[col1, col2]
                if not pd.isna(corr_val):
                    # Ensure we have a numeric value for conversion
                    correlation_dict[col1][col2] = (
                        float(corr_val) if isinstance(corr_val, int | float) else 0.0
                    )
                else:
                    correlation_dict[col1][col2] = 0.0

        # Filter by minimum correlation if specified
        if min_correlation is not None:
            filtered_dict = {}
            for col1, col_corrs in correlation_dict.items():
                filtered_col = {}
                for col2, corr_val in col_corrs.items():
                    if abs(corr_val) >= abs(min_correlation) or col1 == col2:
                        filtered_col[col2] = corr_val
                if filtered_col:
                    filtered_dict[col1] = filtered_col
            correlation_dict = filtered_dict

        session.record_operation(
            OperationType.ANALYZE,
            {
                "type": "correlation",
                "method": method,
                "columns": list(numeric_df.columns),
                "min_correlation": min_correlation,
            },
        )

        return CorrelationResult(
            session_id=session_id,
            method=method,
            correlation_matrix=correlation_dict,
            columns_analyzed=list(numeric_df.columns),
        )

    except (
        SessionNotFoundError,
        NoDataLoadedError,
        ColumnNotFoundError,
        InvalidParameterError,
    ) as e:
        logger.error(f"Correlation calculation failed: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e!s}")
        raise ToolError(f"Error calculating correlation matrix: {e}") from e


async def get_value_counts(
    session_id: str,
    column: str,
    normalize: bool = False,
    sort: bool = True,
    ascending: bool = False,
    top_n: int | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> ValueCountsResult:
    """Get frequency distribution of values in a column.

    Analyzes the distribution of values in a specified column, providing
    counts and optionally percentages for each unique value. Essential for
    understanding categorical data and identifying common patterns.

    Args:
        session_id: Session ID containing loaded data
        column: Name of the column to analyze
        normalize: If True, return percentages instead of counts
        sort: Sort results by frequency
        ascending: Sort in ascending order (default: False for descending)
        top_n: Maximum number of values to return (default: all)
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
                                      normalize=True, top_n=10)

        # Sort in ascending order
        counts = await get_value_counts("session_123", "grade", ascending=True)

    AI Workflow Integration:
        1. Categorical data analysis and encoding decisions
        2. Data quality assessment (identifying rare values)
        3. Understanding distribution for sampling strategies
        4. Feature engineering insights for categorical variables
    """
    try:
        session, df = _get_session_data(session_id)

        if column not in df.columns:
            raise ColumnNotFoundError(column, df.columns.tolist())

        # Get value counts
        value_counts = df[column].value_counts(
            normalize=normalize, sort=sort, ascending=ascending, dropna=True
        )

        # Limit to top_n if specified
        if top_n is not None and top_n > 0:
            value_counts = value_counts.head(top_n)

        # Convert to dict, handling various data types
        counts_dict = {}
        for value, count in value_counts.items():
            # Handle NaN and None values
            if pd.isna(value):
                key = "<null>"
            elif isinstance(value, str | int | float | bool):
                key = str(value)
            else:
                key = str(value)

            counts_dict[key] = float(count) if normalize else int(count)

        # Calculate summary statistics
        total_count = int(df[column].count())  # Non-null count
        unique_count = int(df[column].nunique())

        session.record_operation(
            OperationType.ANALYZE,
            {
                "type": "value_counts",
                "column": column,
                "normalize": normalize,
                "top_n": top_n,
                "unique_values": unique_count,
            },
        )

        return ValueCountsResult(
            session_id=session_id,
            column=column,
            value_counts=counts_dict,
            total_values=total_count,
            unique_values=unique_count,
            normalize=normalize,
        )

    except (SessionNotFoundError, NoDataLoadedError, ColumnNotFoundError) as e:
        logger.error(f"Value counts calculation failed: {e.message}")
        raise ToolError(e.message) from e
    except Exception as e:
        logger.error(f"Error calculating value counts: {e!s}")
        raise ToolError(f"Error calculating value counts: {e}") from e


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
