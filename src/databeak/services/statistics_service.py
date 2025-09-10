"""Statistics service with dependency injection for improved testability.

This module implements statistical operations using the SessionService pattern to reduce coupling
with session management and improve testability.
"""

from __future__ import annotations

import logging
from typing import Literal, cast

import numpy as np
import pandas as pd
from fastmcp.exceptions import ToolError

from ..models.data_models import OperationType
from ..models.session_service import SessionService
from ..models.statistics_models import (
    ColumnStatisticsResult,
    CorrelationResult,
    StatisticsResult,
    StatisticsSummary,
    ValueCountsResult,
)

logger = logging.getLogger(__name__)


class StatisticsService(SessionService):
    """Service for statistical operations with dependency injection."""

    def get_service_name(self) -> str:
        """Get the name of this service."""
        return "StatisticsService"

    async def get_statistics(
        self,
        session_id: str,
        columns: list[str] | None = None,
        include_percentiles: bool = True,
    ) -> StatisticsResult:
        """Get comprehensive statistical summary of numerical columns.

        Args:
            session_id: Session ID containing loaded data
            columns: Optional list of specific columns to analyze
            include_percentiles: Whether to include percentiles

        Returns:
            Comprehensive statistical analysis with per-column summaries
        """
        try:
            session, df = self.get_session_and_data(session_id)

            # Select columns to analyze
            if columns:
                self.validate_columns_exist(df, columns)
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

        except ValueError as e:
            logger.error(f"Validation error in get_statistics: {e}")
            raise ToolError(str(e)) from e
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise ToolError(f"Error getting statistics: {e}") from e

    async def get_column_statistics(
        self,
        session_id: str,
        column: str,
    ) -> ColumnStatisticsResult:
        """Get detailed statistical analysis for a single column.

        Args:
            session_id: Session ID containing loaded data
            column: Name of the column to analyze

        Returns:
            Detailed statistical analysis for the specified column
        """
        try:
            session, df = self.get_session_and_data(session_id)

            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found")

            col_data = df[column]
            col_dtype = str(col_data.dtype)

            # Map pandas dtypes to Pydantic model literals
            dtype_mapping: dict[
                str, Literal["int64", "float64", "object", "bool", "datetime64", "category"]
            ] = {
                "int64": "int64",
                "float64": "float64",
                "bool": "bool",
                "object": "object",
                "category": "category",
            }

            if col_dtype in dtype_mapping:
                mapped_dtype = dtype_mapping[col_dtype]
            elif col_dtype.startswith("datetime64"):
                mapped_dtype = "datetime64"
            else:
                mapped_dtype = "object"

            # Create statistics - only meaningful for numeric columns
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

        except ValueError as e:
            logger.error(f"Validation error in get_column_statistics: {e}")
            raise ToolError(str(e)) from e
        except Exception as e:
            logger.error(f"Column statistics failed: {e}")
            raise ToolError(f"Failed to analyze column '{column}': {e}") from e

    async def get_correlation_matrix(
        self,
        session_id: str,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        columns: list[str] | None = None,
        min_correlation: float | None = None,
    ) -> CorrelationResult:
        """Calculate correlation matrix for numerical columns.

        Args:
            session_id: Session ID containing loaded data
            method: Correlation method
            columns: Optional list of columns to include
            min_correlation: Minimum correlation threshold to include

        Returns:
            Correlation matrix with pairwise correlation coefficients
        """
        try:
            session, df = self.get_session_and_data(session_id)

            # Select columns
            if columns:
                self.validate_columns_exist(df, columns)
                numeric_df = df[columns].select_dtypes(include=[np.number])
            else:
                numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.empty:
                raise ValueError("No numeric columns found")

            if len(numeric_df.columns) < 2:
                raise ValueError("Need at least 2 numeric columns for correlation")

            # Calculate correlation
            if method not in ["pearson", "spearman", "kendall"]:
                raise ValueError(f"Invalid method: {method}")

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

        except ValueError as e:
            logger.error(f"Validation error in get_correlation_matrix: {e}")
            raise ToolError(str(e)) from e
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            raise ToolError(f"Failed to compute correlation matrix: {e}") from e

    async def get_value_counts(
        self,
        session_id: str,
        column: str,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        top_n: int | None = None,
    ) -> ValueCountsResult:
        """Get frequency distribution of values in a column.

        Args:
            session_id: Session ID containing loaded data
            column: Name of the column to analyze
            normalize: If True, return percentages instead of counts
            sort: Sort results by frequency
            ascending: Sort order
            top_n: Maximum number of values to return

        Returns:
            Frequency distribution with counts/percentages for each unique value
        """
        try:
            session, df = self.get_session_and_data(session_id)

            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found")

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

        except ValueError as e:
            logger.error(f"Validation error in get_value_counts: {e}")
            raise ToolError(str(e)) from e
        except Exception as e:
            logger.error(f"Value counts analysis failed: {e}")
            raise ToolError(f"Failed to analyze value counts for column '{column}': {e}") from e
