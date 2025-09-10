"""Standalone Discovery server for DataBeak using FastMCP server composition.

This module provides a complete Discovery server implementation following DataBeak's modular server
architecture pattern. It focuses on data exploration, profiling, pattern detection, and outlier
analysis with specialized algorithms for data insights.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel

# Import session management and data models from the main package
from ..models.tool_responses import BaseToolResponse, CellLocation

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS FOR DISCOVERY OPERATIONS
# ============================================================================


class OutlierInfo(BaseModel):
    """Information about a detected outlier."""

    row_index: int
    value: float
    z_score: float | None = None
    iqr_score: float | None = None


class ProfileInfo(BaseModel):
    """Data profiling information for a column."""

    column_name: str
    data_type: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    most_frequent: Any = None
    frequency: int | None = None


class GroupStatistics(BaseModel):
    """Statistics for a grouped data segment."""

    count: int
    mean: float | None = None
    sum: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None


class OutliersResult(BaseToolResponse):
    """Response model for outlier detection analysis."""

    session_id: str
    outliers_found: int
    outliers_by_column: dict[str, list[OutlierInfo]]
    method: Literal["zscore", "iqr", "isolation_forest"]
    threshold: float


class ProfileResult(BaseToolResponse):
    """Response model for comprehensive data profiling."""

    session_id: str
    profile: dict[str, ProfileInfo]
    total_rows: int
    total_columns: int
    memory_usage_mb: float
    include_correlations: bool = True
    include_outliers: bool = True


class GroupAggregateResult(BaseToolResponse):
    """Response model for group aggregation operations."""

    session_id: str
    groups: dict[str, dict[str, GroupStatistics]]
    group_by_columns: list[str]
    aggregated_columns: list[str]
    total_groups: int


class DataSummaryResult(BaseToolResponse):
    """Response model for data overview and summary."""

    session_id: str
    summary: dict[str, Any]
    shape: tuple[int, int]
    column_info: dict[str, str]
    memory_usage_mb: float


class FindCellsResult(BaseToolResponse):
    """Response model for cell value search operations."""

    session_id: str
    search_value: Any
    matches: list[CellLocation]
    total_matches: int
    columns_searched: list[str]
    exact_match: bool


class InspectDataResult(BaseToolResponse):
    """Response model for contextual data inspection."""

    session_id: str
    center_coordinates: dict[str, Any]
    surrounding_data: dict[str, Any]
    radius: int
    data_preview: dict[str, Any]


# ============================================================================
# DISCOVERY OPERATIONS LOGIC
# ============================================================================


async def detect_outliers(
    session_id: str,
    columns: list[str],
    method: Literal["zscore", "iqr", "isolation_forest"] = "zscore",
    threshold: float = 3.0,
    ctx: Context | None = None,
) -> OutliersResult:
    """Detect outliers in numerical columns using various algorithms.

    Identifies data points that deviate significantly from the normal pattern
    using statistical and machine learning methods. Essential for data quality
    assessment and anomaly detection in analytical workflows.

    Args:
        session_id: Session ID containing loaded data
        columns: List of numerical columns to analyze for outliers
        method: Detection algorithm (zscore, iqr, isolation_forest)
        threshold: Sensitivity threshold (higher = less sensitive)
        ctx: FastMCP context for progress reporting

    Returns:
        Detailed outlier analysis with locations and severity scores

    Detection Methods:
        📊 Z-Score: Statistical method based on standard deviations
        📈 IQR: Interquartile range method (robust to distribution)
        🤖 Isolation Forest: ML-based method for high-dimensional data

    Examples:
        # Basic outlier detection
        outliers = await detect_outliers("session_123", ["price", "quantity"])

        # Use IQR method with custom threshold
        outliers = await detect_outliers("session_123", ["sales"],
                                        method="iqr", threshold=2.5)

    AI Workflow Integration:
        1. Data quality assessment and cleaning
        2. Anomaly detection for fraud/error identification
        3. Data preprocessing for machine learning
        4. Understanding data distribution characteristics
    """
    try:
        from ..tools.analytics import detect_outliers as _detect_outliers

        return await _detect_outliers(session_id, columns, method, threshold, ctx)

    except Exception as e:
        logger.error(f"Outlier detection failed: {e}")
        if ctx:
            await ctx.error(f"Outlier detection failed: {e!s}")
        raise ToolError(f"Failed to detect outliers: {e}") from e


async def profile_data(
    session_id: str,
    include_correlations: bool = True,
    include_outliers: bool = True,
    ctx: Context | None = None,
) -> ProfileResult:
    """Generate comprehensive data profile with statistical insights.

    Creates a complete analytical profile of the dataset including column
    characteristics, data types, null patterns, correlations, and outliers.
    Provides holistic data understanding for analytical workflows.

    Args:
        session_id: Session ID containing loaded data
        include_correlations: Include correlation analysis in profile
        include_outliers: Include outlier detection in profile
        ctx: FastMCP context for progress reporting

    Returns:
        Comprehensive data profile with multi-dimensional analysis

    Profile Components:
        📊 Column Profiles: Data types, null patterns, uniqueness
        📈 Statistical Summaries: Numerical column characteristics
        🔗 Correlations: Inter-variable relationships (optional)
        🎯 Outliers: Anomaly detection across columns (optional)
        💾 Memory Usage: Resource consumption analysis

    Examples:
        # Full data profile
        profile = await profile_data("session_123")

        # Quick profile without expensive computations
        profile = await profile_data("session_123",
                                   include_correlations=False,
                                   include_outliers=False)

    AI Workflow Integration:
        1. Initial data exploration and understanding
        2. Automated data quality reporting
        3. Feature engineering guidance
        4. Data preprocessing strategy development
    """
    try:
        from ..tools.analytics import profile_data as _profile_data

        return await _profile_data(session_id, include_correlations, include_outliers, ctx)

    except Exception as e:
        logger.error(f"Data profiling failed: {e}")
        if ctx:
            await ctx.error(f"Data profiling failed: {e!s}")
        raise ToolError(f"Failed to profile data: {e}") from e


async def group_by_aggregate(
    session_id: str,
    group_by: list[str],
    aggregations: dict[str, list[str]],
    ctx: Context | None = None,
) -> GroupAggregateResult:
    """Group data and compute aggregations for analytical insights.

    Performs SQL-like GROUP BY operations with multiple aggregation functions
    per column. Essential for segmentation analysis and understanding patterns
    across different data groups.

    Args:
        session_id: Session ID containing loaded data
        group_by: List of columns to group by
        aggregations: Dict mapping column names to list of aggregation functions
        ctx: FastMCP context for progress reporting

    Returns:
        Grouped aggregation results with statistics per group

    Aggregation Functions:
        📊 count, mean, median, sum, min, max
        📈 std, var (statistical measures)
        🎯 first, last (positional)
        📋 nunique (unique count)

    Examples:
        # Sales analysis by region
        result = await group_by_aggregate("session_123",
                                        group_by=["region"],
                                        aggregations={"sales": ["sum", "mean", "count"]})

        # Multi-dimensional grouping
        result = await group_by_aggregate("session_123",
                                        group_by=["category", "region"],
                                        aggregations={
                                            "price": ["mean", "std"],
                                            "quantity": ["sum", "count"]
                                        })

    AI Workflow Integration:
        1. Segmentation analysis and market research
        2. Feature engineering for categorical interactions
        3. Data summarization for reporting and insights
        4. Understanding group-based patterns and trends
    """
    try:
        from ..tools.analytics import group_by_aggregate as _group_by_aggregate

        return await _group_by_aggregate(session_id, group_by, aggregations, ctx)

    except Exception as e:
        logger.error(f"Group aggregation failed: {e}")
        if ctx:
            await ctx.error(f"Group aggregation failed: {e!s}")
        raise ToolError(f"Failed to perform group aggregation: {e}") from e


async def find_cells_with_value(
    session_id: str,
    value: Any,
    columns: list[str] | None = None,
    exact_match: bool = True,
    ctx: Context | None = None,
) -> FindCellsResult:
    """Find all cells containing a specific value for data discovery.

    Searches through the dataset to locate all occurrences of a specific value,
    providing coordinates and context. Essential for data validation, quality
    checking, and understanding data patterns.

    Args:
        session_id: Session ID containing loaded data
        value: The value to search for (any data type)
        columns: Optional list of columns to search (default: all)
        exact_match: If True, require exact match; if False, substring search
        ctx: FastMCP context for progress reporting

    Returns:
        Locations of all matching cells with coordinates and context

    Search Features:
        🎯 Exact Match: Precise value matching with type consideration
        🔍 Substring Search: Flexible text-based search for string columns
        📍 Coordinates: Row and column positions for each match
        📊 Summary Stats: Total matches, columns searched, search parameters

    Examples:
        # Find all cells with value "ERROR"
        results = await find_cells_with_value("session_123", "ERROR")

        # Substring search in specific columns
        results = await find_cells_with_value("session_123", "john",
                                            columns=["name", "email"],
                                            exact_match=False)

    AI Workflow Integration:
        1. Data quality assessment and error detection
        2. Pattern identification and data validation
        3. Reference data location and verification
        4. Data cleaning and preprocessing guidance
    """
    try:
        from ..tools.transformations import find_cells_with_value as _find_cells_with_value

        return await _find_cells_with_value(session_id, value, columns, exact_match, ctx)

    except Exception as e:
        logger.error(f"Cell search failed: {e}")
        if ctx:
            await ctx.error(f"Cell search failed: {e!s}")
        raise ToolError(f"Failed to search for value '{value}': {e}") from e


async def get_data_summary(
    session_id: str,
    include_preview: bool = True,
    ctx: Context | None = None,
) -> DataSummaryResult:
    """Get comprehensive data overview and structural summary.

    Provides high-level overview of dataset structure, dimensions, data types,
    and memory usage. Essential first step in data exploration and analysis
    planning workflows.

    Args:
        session_id: Session ID containing loaded data
        include_preview: Include sample data rows in summary
        ctx: FastMCP context for progress reporting

    Returns:
        Comprehensive data overview with structural information

    Summary Components:
        📏 Dimensions: Rows, columns, shape information
        🔢 Data Types: Column type distribution and analysis
        💾 Memory Usage: Resource consumption breakdown
        👀 Preview: Sample rows for quick data understanding (optional)
        📊 Overview: High-level dataset characteristics

    Examples:
        # Full data summary with preview
        summary = await get_data_summary("session_123")

        # Structure summary without preview data
        summary = await get_data_summary("session_123", include_preview=False)

    AI Workflow Integration:
        1. Initial data exploration and understanding
        2. Planning analytical approaches based on data structure
        3. Resource planning for large dataset processing
        4. Data quality initial assessment
    """
    try:
        from ..tools.transformations import get_data_summary as _get_data_summary

        return await _get_data_summary(session_id, include_preview, ctx)

    except Exception as e:
        logger.error(f"Data summary failed: {e}")
        if ctx:
            await ctx.error(f"Data summary failed: {e!s}")
        raise ToolError(f"Failed to generate data summary: {e}") from e


async def inspect_data_around(
    session_id: str,
    row: int,
    column_name: str,
    radius: int = 2,
    ctx: Context | None = None,
) -> InspectDataResult:
    """Inspect data around a specific coordinate for contextual analysis.

    Examines the data surrounding a specific cell to understand context,
    patterns, and relationships. Useful for data validation, error investigation,
    and understanding local data patterns.

    Args:
        session_id: Session ID containing loaded data
        row: Row index to center the inspection (0-based)
        column_name: Name of the column to center on
        radius: Number of rows/columns to include around center point
        ctx: FastMCP context for progress reporting

    Returns:
        Contextual view of data around the specified coordinates

    Inspection Features:
        📍 Center Point: Specified cell as reference point
        🔍 Radius View: Configurable area around center cell
        📊 Data Context: Surrounding values for pattern analysis
        🎯 Coordinates: Clear row/column reference system

    Examples:
        # Inspect around a specific data point
        context = await inspect_data_around("session_123", row=50,
                                          column_name="price", radius=3)

        # Minimal context view
        context = await inspect_data_around("session_123", row=10,
                                          column_name="status", radius=1)

    AI Workflow Integration:
        1. Error investigation and data quality assessment
        2. Pattern recognition in local data areas
        3. Understanding data relationships and context
        4. Validation of data transformations and corrections
    """
    try:
        from ..tools.transformations import inspect_data_around as _inspect_data_around

        return await _inspect_data_around(session_id, row, column_name, radius, ctx)

    except Exception as e:
        logger.error(f"Data inspection failed: {e}")
        if ctx:
            await ctx.error(f"Data inspection failed: {e!s}")
        raise ToolError(
            f"Failed to inspect data around row {row}, column '{column_name}': {e}"
        ) from e


# ============================================================================
# FASTMCP SERVER SETUP
# ============================================================================


# Create Discovery server
discovery_server = FastMCP(
    "DataBeak-Discovery",
    instructions="Data exploration and profiling server for DataBeak with comprehensive discovery and pattern detection capabilities",
)


# Register the data discovery and profiling functions directly as MCP tools
discovery_server.tool(name="detect_outliers")(detect_outliers)
discovery_server.tool(name="profile_data")(profile_data)
discovery_server.tool(name="group_by_aggregate")(group_by_aggregate)
discovery_server.tool(name="find_cells_with_value")(find_cells_with_value)
discovery_server.tool(name="get_data_summary")(get_data_summary)
discovery_server.tool(name="inspect_data_around")(inspect_data_around)
