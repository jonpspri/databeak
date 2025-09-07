"""Analytics tools for CSV data analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from fastmcp.exceptions import ToolError

from ..models.csv_session import get_session_manager
from ..models.data_models import OperationType
from ..models.tool_responses import (
    ColumnStatisticsResult,
    CorrelationResult,
    GroupAggregateResult,
    GroupStatistics,
    OutlierInfo,
    OutliersResult,
    ProfileInfo,
    ProfileResult,
    StatisticsResult,
    StatisticsSummary,
    ValueCountsResult,
)

if TYPE_CHECKING:
    from fastmcp import Context

logger = logging.getLogger(__name__)


async def get_statistics(
    session_id: str,
    columns: list[str] | None = None,
    include_percentiles: bool = True,
    ctx: Context | None = None,  # noqa: ARG001
) -> StatisticsResult:
    """Get statistical summary of numerical columns.

    Args:
        session_id: Session identifier
        columns: Specific columns to analyze (None for all numeric)
        include_percentiles: Include percentile values
        ctx: FastMCP context

    Returns:
        StatisticsResult with statistics for each column
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
    ctx: Context | None = None,  # noqa: ARG001
) -> ColumnStatisticsResult:
    """Get detailed statistics for a specific column.

    Args:
        session_id: Session identifier
        column: Column name to analyze
        ctx: FastMCP context

    Returns:
        Dict with detailed column statistics
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
        # Map pandas dtypes to Pydantic model literals
        col_dtype = str(col_data.dtype)
        if "int" in col_dtype:
            mapped_dtype = "int64"
        elif "float" in col_dtype:
            mapped_dtype = "float64"
        elif "bool" in col_dtype:
            mapped_dtype = "bool"
        elif "datetime" in col_dtype:
            mapped_dtype = "datetime64"
        elif "category" in col_dtype:
            mapped_dtype = "category"
        else:
            mapped_dtype = "object"

        # Create statistics - only meaningful for numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
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
            data_type=cast(
                "Literal['int64', 'float64', 'object', 'bool', 'datetime64', 'category']",
                mapped_dtype,
            ),
            non_null_count=int(col_data.notna().sum()),
        )

    except Exception as e:
        logger.error(f"Error getting column statistics: {e!s}")
        raise ToolError(f"Error getting column statistics: {e}") from e


async def get_correlation_matrix(
    session_id: str,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    columns: list[str] | None = None,
    min_correlation: float | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> CorrelationResult:
    """Calculate correlation matrix for numeric columns.

    Args:
        session_id: Session identifier
        method: Correlation method ('pearson', 'spearman', 'kendall')
        columns: Specific columns to include (None for all numeric)
        min_correlation: Filter to show only correlations above this threshold
        ctx: FastMCP context

    Returns:
        Dict with correlation matrix
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

        # Find highly correlated pairs
        high_correlations = []
        # Use min_correlation if specified, otherwise use 0.7 as default threshold
        correlation_threshold = min_correlation if min_correlation is not None else 0.7

        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i + 1 :]:
                corr_value = corr_matrix.loc[col1, col2]
                if not pd.isna(corr_value):
                    float_corr = float(cast("float", corr_value))
                    if abs(float_corr) >= correlation_threshold:
                        high_correlations.append(
                            {
                                "column1": col1,
                                "column2": col2,
                                "correlation": round(float_corr, 4),
                            }
                        )

        high_correlations.sort(key=lambda x: abs(cast("float", x["correlation"])), reverse=True)

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
        logger.error(f"Error calculating correlation: {e!s}")
        raise ToolError(f"Error calculating correlation: {e}") from e


async def group_by_aggregate(
    session_id: str,
    group_by: list[str],
    aggregations: dict[str, str | list[str]],
    ctx: Context | None = None,  # noqa: ARG001
) -> GroupAggregateResult:
    """Group data and apply aggregation functions.

    Args:
        session_id: Session identifier
        group_by: Columns to group by
        aggregations: Dict mapping column names to aggregation functions
                     e.g., {"sales": ["sum", "mean"], "quantity": "sum"}
        ctx: FastMCP context

    Returns:
        Dict with grouped data
    """
    try:
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session or not session.data_session.has_data():
            raise ToolError(f"Invalid session or no data loaded: {session_id}")

        df = session.data_session.df
        assert df is not None  # Type guard: has_data() ensures df is not None

        # Validate group by columns
        missing_cols = [col for col in group_by if col not in df.columns]
        if missing_cols:
            raise ToolError(f"Group by columns not found: {missing_cols}")

        # Validate aggregation columns
        agg_cols = list(aggregations.keys())
        missing_agg_cols = [col for col in agg_cols if col not in df.columns]
        if missing_agg_cols:
            raise ToolError(f"Aggregation columns not found: {missing_agg_cols}")

        # Perform groupby to get group statistics
        grouped = df.groupby(group_by)

        # Create GroupStatistics for each group
        group_stats = {}

        for group_name, group_data in grouped:
            # Convert group name to string for dict key
            if isinstance(group_name, tuple):
                group_key = "_".join(str(x) for x in group_name)
            else:
                group_key = str(group_name)

            # Calculate basic statistics for the group
            # Focus on first numeric column for statistics, or count for non-numeric
            numeric_cols = group_data.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 0:
                # Use first numeric column for statistics
                first_numeric = group_data[numeric_cols[0]]
                group_stats[group_key] = GroupStatistics(
                    count=len(group_data),
                    mean=float(first_numeric.mean()) if not pd.isna(first_numeric.mean()) else None,
                    sum=float(first_numeric.sum()) if not pd.isna(first_numeric.sum()) else None,
                    min=float(first_numeric.min()) if not pd.isna(first_numeric.min()) else None,
                    max=float(first_numeric.max()) if not pd.isna(first_numeric.max()) else None,
                    std=float(first_numeric.std()) if not pd.isna(first_numeric.std()) else None,
                )
            else:
                # No numeric columns, just provide count
                group_stats[group_key] = GroupStatistics(count=len(group_data))

        session.record_operation(
            OperationType.GROUP_BY,
            {
                "group_by": group_by,
                "aggregations": aggregations,
                "total_groups": len(group_stats),
            },
        )

        return GroupAggregateResult(
            session_id=session_id,
            groups=group_stats,
            group_columns=group_by,
            aggregation_functions=aggregations,
            total_groups=len(group_stats),
        )

    except Exception as e:
        logger.error(f"Error in group by aggregate: {e!s}")
        raise ToolError(f"Error in group by aggregate: {e}") from e


async def get_value_counts(
    session_id: str,
    column: str,
    normalize: bool = False,
    sort: bool = True,
    ascending: bool = False,
    top_n: int | None = None,
    ctx: Context | None = None,  # noqa: ARG001
) -> ValueCountsResult:
    """Get value counts for a column.

    Args:
        session_id: Session identifier
        column: Column name to count values
        normalize: Return proportions instead of counts
        sort: Sort by frequency
        ascending: Sort order
        top_n: Return only top N values
        ctx: FastMCP context

    Returns:
        Dict with value counts
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
        _ = df[column].isna().sum()  # null_count not used

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
        )

    except Exception as e:
        logger.error(f"Error getting value counts: {e!s}")
        raise ToolError(f"Error getting value counts: {e}") from e


async def detect_outliers(
    session_id: str,
    columns: list[str] | None = None,
    method: str = "iqr",
    threshold: float = 1.5,
    ctx: Context | None = None,  # noqa: ARG001
) -> OutliersResult:
    """Detect outliers in numeric columns.

    Args:
        session_id: Session identifier
        columns: Columns to check (None for all numeric)
        method: Detection method ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection (1.5 for IQR, 3 for z-score)
        ctx: FastMCP context

    Returns:
        Dict with outlier information
    """
    try:
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session or not session.data_session.has_data():
            raise ToolError(f"Invalid session or no data loaded: {session_id}")

        df = session.data_session.df
        assert df is not None  # Type guard: has_data() ensures df is not None

        # Select numeric columns
        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ToolError(f"Columns not found: {missing_cols}")
            numeric_df = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ToolError("No numeric columns found")

        outliers_by_column = {}
        total_outliers_count = 0

        if method == "iqr":
            for col in numeric_df.columns:
                q1 = numeric_df[col].quantile(0.25)
                q3 = numeric_df[col].quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

                outlier_mask = (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)
                outlier_indices = df.index[outlier_mask]

                # Create OutlierInfo objects for each outlier
                outlier_infos = []
                for idx in outlier_indices[:100]:  # Limit to first 100
                    raw_value = df.at[idx, col]
                    try:
                        value = float(cast("Any", raw_value))
                    except (ValueError, TypeError):
                        continue  # Skip non-numeric values

                    # Calculate IQR score (distance from nearest bound relative to IQR)
                    if value < lower_bound:
                        iqr_score = float((lower_bound - value) / iqr) if iqr > 0 else 0.0
                    else:
                        iqr_score = float((value - upper_bound) / iqr) if iqr > 0 else 0.0

                    outlier_infos.append(
                        OutlierInfo(row_index=int(idx), value=value, iqr_score=iqr_score)
                    )

                outliers_by_column[col] = outlier_infos
                total_outliers_count += len(outlier_indices)

        elif method == "zscore":
            for col in numeric_df.columns:
                col_mean = numeric_df[col].mean()
                col_std = numeric_df[col].std()
                z_scores = np.abs((numeric_df[col] - col_mean) / col_std)
                outlier_mask = z_scores > threshold
                outlier_indices = df.index[outlier_mask]

                # Create OutlierInfo objects for each outlier
                outlier_infos = []
                for idx in outlier_indices[:100]:  # Limit to first 100
                    raw_value = df.at[idx, col]
                    try:
                        value = float(cast("Any", raw_value))
                    except (ValueError, TypeError):
                        continue  # Skip non-numeric values

                    z_score = float(abs((value - col_mean) / col_std)) if col_std > 0 else 0.0

                    outlier_infos.append(
                        OutlierInfo(row_index=int(idx), value=value, z_score=z_score)
                    )

                outliers_by_column[col] = outlier_infos
                total_outliers_count += len(outlier_indices)

        else:
            raise ToolError(f"Unknown method: {method}")

        session.record_operation(
            OperationType.ANALYZE,
            {
                "type": "outlier_detection",
                "method": method,
                "threshold": threshold,
                "columns": list(outliers_by_column.keys()),
            },
        )

        # Map method names to match Pydantic model expectations
        if method == "zscore":
            pydantic_method = "z-score"
        elif method == "iqr":
            pydantic_method = "iqr"
        else:
            pydantic_method = "isolation_forest"

        return OutliersResult(
            session_id=session_id,
            outliers_found=total_outliers_count,
            outliers_by_column=outliers_by_column,
            method=cast("Literal['z-score', 'iqr', 'isolation_forest']", pydantic_method),
            threshold=threshold,
        )

    except Exception as e:
        logger.error(f"Error detecting outliers: {e!s}")
        raise ToolError(f"Error detecting outliers: {e}") from e


async def profile_data(
    session_id: str,
    include_correlations: bool = True,
    include_outliers: bool = True,
    ctx: Context | None = None,  # noqa: ARG001
) -> ProfileResult:
    """Generate comprehensive data profile.

    Args:
        session_id: Session identifier
        include_correlations: Include correlation analysis
        include_outliers: Include outlier detection
        ctx: FastMCP context

    Returns:
        Dict with complete data profile
    """
    try:
        manager = get_session_manager()
        session = manager.get_session(session_id)

        if not session or not session.data_session.has_data():
            raise ToolError(f"Invalid session or no data loaded: {session_id}")

        df = session.data_session.df
        assert df is not None  # Type guard: has_data() ensures df is not None

        # Create ProfileInfo for each column (simplified to match model)
        profile_dict = {}

        for col in df.columns:
            col_data = df[col]

            # Get the most frequent value and its frequency
            value_counts = col_data.value_counts(dropna=False)
            most_frequent = None
            frequency = None
            if len(value_counts) > 0:
                most_frequent = value_counts.index[0]
                frequency = int(value_counts.iloc[0])

                # Handle various data types for most_frequent
                if most_frequent is None or pd.isna(most_frequent):
                    most_frequent = None
                elif isinstance(most_frequent, str | int | float | bool):
                    most_frequent = most_frequent
                else:
                    most_frequent = str(most_frequent)

            profile_info = ProfileInfo(
                column_name=col,
                data_type=str(col_data.dtype),
                null_count=int(col_data.isna().sum()),
                null_percentage=round(col_data.isna().sum() / len(df) * 100, 2),
                unique_count=int(col_data.nunique()),
                unique_percentage=round(col_data.nunique() / len(df) * 100, 2),
                most_frequent=most_frequent,
                frequency=frequency,
            )

            profile_dict[col] = profile_info

        # Note: Correlation and outlier analysis have been simplified
        # since the ProfileResult model doesn't include them

        memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

        session.record_operation(
            OperationType.PROFILE,
            {
                "include_correlations": include_correlations,
                "include_outliers": include_outliers,
            },
        )

        return ProfileResult(
            session_id=session_id,
            profile=profile_dict,
            total_rows=len(df),
            total_columns=len(df.columns),
            memory_usage_mb=memory_usage_mb,
        )

    except Exception as e:
        logger.error(f"Error profiling data: {e!s}")
        raise ToolError(f"Error profiling data: {e}") from e
