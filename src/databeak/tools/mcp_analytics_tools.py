"""FastMCP analytics tool definitions for DataBeak."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from fastmcp import Context, FastMCP

from .analytics import detect_outliers as _detect_outliers
from .analytics import get_column_statistics as _get_column_statistics
from .analytics import get_correlation_matrix as _get_correlation_matrix
from .analytics import get_statistics as _get_statistics
from .analytics import get_value_counts as _get_value_counts
from .analytics import group_by_aggregate as _group_by_aggregate
from .analytics import profile_data as _profile_data

# Import type aliases
from .transformations import CellValue
from .transformations import find_cells_with_value as _find_cells_with_value
from .transformations import get_data_summary as _get_data_summary
from .transformations import inspect_data_around as _inspect_data_around

if TYPE_CHECKING:
    from ..models.tool_responses import (
        ColumnStatisticsResult,
        CorrelationResult,
        DataSummaryResult,
        FindCellsResult,
        GroupAggregateResult,
        InspectDataResult,
        OutliersResult,
        ProfileResult,
        StatisticsResult,
        ValueCountsResult,
    )


def register_analytics_tools(mcp: FastMCP) -> None:
    """Register analytics tools with FastMCP server."""

    @mcp.tool
    async def get_statistics(
        session_id: str,
        columns: list[str] | None = None,
        include_percentiles: bool = True,
        ctx: Context | None = None,
    ) -> StatisticsResult:
        """Get statistical summary of numerical columns."""
        return await _get_statistics(session_id, columns, include_percentiles, ctx)

    @mcp.tool
    async def get_column_statistics(
        session_id: str, column: str, ctx: Context | None = None
    ) -> ColumnStatisticsResult:
        """Get detailed statistics for a specific column."""
        return await _get_column_statistics(session_id, column, ctx)

    @mcp.tool
    async def get_correlation_matrix(
        session_id: str,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        columns: list[str] | None = None,
        min_correlation: float | None = None,
        ctx: Context | None = None,
    ) -> CorrelationResult:
        """Calculate correlation matrix for numeric columns."""
        return await _get_correlation_matrix(session_id, method, columns, min_correlation, ctx)

    @mcp.tool
    async def group_by_aggregate(
        session_id: str,
        group_by: list[str],
        aggregations: dict[str, str | list[str]],
        ctx: Context | None = None,
    ) -> GroupAggregateResult:
        """Group data and apply aggregation functions."""
        return await _group_by_aggregate(session_id, group_by, aggregations, ctx)

    @mcp.tool
    async def get_value_counts(
        session_id: str,
        column: str,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        top_n: int | None = None,
        ctx: Context | None = None,
    ) -> ValueCountsResult:
        """Get value counts for a column."""
        return await _get_value_counts(session_id, column, normalize, sort, ascending, top_n, ctx)

    @mcp.tool
    async def detect_outliers(
        session_id: str,
        columns: list[str] | None = None,
        method: str = "iqr",
        threshold: float = 1.5,
        ctx: Context | None = None,
    ) -> OutliersResult:
        """Detect outliers in numeric columns."""
        return await _detect_outliers(session_id, columns, method, threshold, ctx)

    @mcp.tool
    async def profile_data(
        session_id: str,
        include_correlations: bool = True,
        include_outliers: bool = True,
        ctx: Context | None = None,
    ) -> ProfileResult:
        """Generate comprehensive data profile."""
        return await _profile_data(session_id, include_correlations, include_outliers, ctx)

    # AI-Friendly convenience tools
    @mcp.tool
    async def inspect_data_around(
        session_id: str,
        row: int,
        column: str | int,
        radius: int = 2,
        ctx: Context | None = None,
    ) -> InspectDataResult:
        """Get data around a specific cell for context inspection.

        Args:
            session_id: Session identifier
            row: Center row index (0-based)
            column: Column name (str) or column index (int, 0-based)
            radius: Number of rows/columns around the center to include

        Returns:
            Dict with surrounding data and coordinate information

        Example:
            inspect_data_around("session123", 5, "name", 2) -> Get 5x5 grid centered on (5, "name")
        """
        return await _inspect_data_around(session_id, row, column, radius, ctx)

    @mcp.tool
    async def find_cells_with_value(
        session_id: str,
        value: CellValue,
        column: str | None = None,
        exact_match: bool = True,
        ctx: Context | None = None,
    ) -> FindCellsResult:
        """Find all cells containing a specific value.

        Args:
            session_id: Session identifier
            value: Value to search for
            column: Optional column name to restrict search (None for all columns)
            exact_match: Whether to use exact matching or substring matching for strings

        Returns:
            Dict with coordinates of matching cells

        Examples:
            find_cells_with_value("session123", "John") -> Find all cells with "John"
            find_cells_with_value("session123", 25, "age") -> Find all age cells with value 25
            find_cells_with_value("session123", "john", None, False) -> Substring search across all columns
        """
        return await _find_cells_with_value(session_id, value, column, exact_match, ctx)

    @mcp.tool
    async def get_data_summary(
        session_id: str,
        include_preview: bool = True,
        max_preview_rows: int = 10,
        ctx: Context | None = None,
    ) -> DataSummaryResult:
        """Get comprehensive data summary optimized for AI understanding and workflow planning.

        This is the primary tool for AI assistants to understand CSV data structure, content patterns,
        and coordinate system. Provides all essential metadata needed for intelligent data manipulation.

        Args:
            session_id: Session identifier for the active CSV data session
            include_preview: Whether to include sample data preview (recommended: True)
            max_preview_rows: Maximum number of preview rows to include (default: 10)

        Returns:
            Comprehensive analysis containing:
            - success: bool operation status
            - session_id: Session identifier for continued operations
            - coordinate_system: Complete indexing documentation for AI reference
            - shape: {"rows": N, "columns": M} dimensions
            - columns: {"names": [...], "types": {...}, "count": N}
            - data_types: {"numeric_columns": [...], "text_columns": [...], "datetime_columns": [...]}
            - missing_data: {"total_nulls": N, "null_by_column": {...}, "null_percentage": {...}}
            - memory_usage_mb: Memory consumption information
            - preview: {"records": [...], "total_rows": N, "preview_rows": M} with __row_index__

        Key Features:
            ✅ Enhanced preview includes __row_index__ field for coordinate reference
            ✅ Complete column type analysis for informed transformations
            ✅ Null value distribution analysis for data quality assessment
            ✅ Memory usage tracking for performance optimization

        Examples:
            # Start every AI workflow with comprehensive summary
            get_data_summary("session123")

            # Quick overview without sample data
            get_data_summary("session123", False)

            # Large preview for detailed analysis
            get_data_summary("session123", True, 20)

        AI Workflow Integration:
            1. **Start Here**: Always begin data work with get_data_summary()
            2. **Plan Operations**: Use shape/columns info to plan transformations
            3. **Drill Down**: Use preview data to identify specific rows/cells for get_cell_value()
            4. **Quality Check**: Use missing_data analysis to plan null handling
            5. **Next Steps**: Use coordinate_system info for precise operations

        Related Tools:
            → get_session_info(): Basic session metadata
            → get_row_data(): Detailed row inspection
            → get_cell_value(): Individual cell access
            → get_column_data(): Column-specific analysis
        """
        return await _get_data_summary(session_id, include_preview, max_preview_rows, ctx)
