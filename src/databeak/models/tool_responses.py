"""Unified Pydantic response models for all MCP tools.

This module consolidates all tool response models to eliminate type conflicts and provide consistent
structured responses across DataBeak's MCP interface.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Type alias for CSV cell values - more specific than Any while still flexible
CsvCellValue = str | int | float | bool | None

# =============================================================================
# NESTED PYDANTIC MODELS FOR STRUCTURED DATA
# =============================================================================


class SessionInfo(BaseModel):
    """Session information in list results."""

    session_id: str
    created_at: str
    last_accessed: str
    row_count: int
    column_count: int
    columns: list[str]
    memory_usage_mb: float
    file_path: str | None = None


class OutlierInfo(BaseModel):
    """Outlier information for analytics results."""

    row_index: int
    value: float
    z_score: float | None = None
    iqr_score: float | None = None


class StatisticsSummary(BaseModel):
    """Column statistics summary."""

    count: int
    mean: float
    std: float
    min: float
    percentile_25: float = Field(alias="25%")
    percentile_50: float = Field(alias="50%")
    percentile_75: float = Field(alias="75%")
    max: float

    model_config = ConfigDict(populate_by_name=True)


class DataTypeInfo(BaseModel):
    """Data type information for columns."""

    type: Literal["int64", "float64", "object", "bool", "datetime64", "category"]
    nullable: bool
    unique_count: int
    null_count: int


class MissingDataInfo(BaseModel):
    """Missing data summary."""

    total_missing: int
    missing_by_column: dict[str, int]
    missing_percentage: float


class DataPreview(BaseModel):
    """Data preview with row samples."""

    rows: list[dict[str, CsvCellValue]]  # Row data structure varies by dataset but values are typed
    row_count: int
    column_count: int
    truncated: bool = False


class GroupStatistics(BaseModel):
    """Grouped aggregation statistics."""

    count: int
    mean: float | None = None
    sum: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None


class CellLocation(BaseModel):
    """Cell location and value information."""

    row: int
    column: str
    value: CsvCellValue  # CSV cells can contain str, int, float, bool, or None


class ProfileInfo(BaseModel):
    """Data profiling information."""

    column_name: str
    data_type: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    most_frequent: CsvCellValue = None  # Most frequent value can be any CSV data type
    frequency: int | None = None


class BaseToolResponse(BaseModel):
    """Base response model for all MCP tool operations."""

    success: bool = True


# =============================================================================
# SYSTEM TOOL RESPONSES
# =============================================================================


class HealthResult(BaseToolResponse):
    """Response model for system health check."""

    status: str
    version: str
    active_sessions: int
    max_sessions: int
    session_ttl_minutes: int


class ServerInfoResult(BaseToolResponse):
    """Response model for server information and capabilities."""

    name: str
    version: str
    description: str
    capabilities: dict[str, list[str]]
    supported_formats: list[str]
    max_file_size_mb: int
    session_timeout_minutes: int


# =============================================================================
# ANALYTICS TOOL RESPONSES
# =============================================================================


class StatisticsResult(BaseToolResponse):
    """Response model for statistical analysis operations."""

    session_id: str
    statistics: dict[str, StatisticsSummary]
    column_count: int
    numeric_columns: list[str]
    total_rows: int


class CorrelationResult(BaseToolResponse):
    """Response model for correlation matrix operations."""

    session_id: str
    correlation_matrix: dict[str, dict[str, float]]
    method: Literal["pearson", "spearman", "kendall"]
    columns_analyzed: list[str]


class ValueCountsResult(BaseToolResponse):
    """Response model for value counts operations."""

    session_id: str
    column: str
    value_counts: dict[str, int | float]
    total_values: int
    unique_values: int


class OutliersResult(BaseToolResponse):
    """Response model for outlier detection operations."""

    session_id: str
    outliers_found: int
    outliers_by_column: dict[str, list[OutlierInfo]]
    method: Literal["z-score", "iqr", "isolation_forest"]
    threshold: float


class ProfileResult(BaseToolResponse):
    """Response model for comprehensive data profiling."""

    session_id: str
    profile: dict[str, ProfileInfo]
    total_rows: int
    total_columns: int
    memory_usage_mb: float


class DataSummaryResult(BaseToolResponse):
    """Response model for comprehensive data summary."""

    session_id: str
    coordinate_system: dict[str, str]
    shape: dict[str, int]
    columns: dict[str, DataTypeInfo]
    data_types: dict[str, list[str]]
    missing_data: MissingDataInfo
    memory_usage_mb: float
    preview: DataPreview | None = None


class ColumnStatisticsResult(BaseToolResponse):
    """Response model for column-specific statistics."""

    session_id: str
    column: str
    statistics: StatisticsSummary
    data_type: Literal["int64", "float64", "object", "bool", "datetime64", "category"]
    non_null_count: int


class GroupAggregateResult(BaseToolResponse):
    """Response model for group-by aggregation operations."""

    session_id: str
    groups: dict[str, GroupStatistics]
    group_columns: list[str]
    aggregation_functions: dict[str, str | list[str]]
    total_groups: int


class InspectDataResult(BaseToolResponse):
    """Response model for data inspection around specific coordinates."""

    session_id: str
    center_coordinates: dict[str, str | int]
    surrounding_data: DataPreview
    radius: int


class FindCellsResult(BaseToolResponse):
    """Response model for cell search operations."""

    session_id: str
    search_value: str | int | float | bool | None
    matches_found: int
    coordinates: list[CellLocation]
    search_column: str | None
    exact_match: bool


# =============================================================================
# ROW TOOL RESPONSES
# =============================================================================


class CellValueResult(BaseToolResponse):
    """Response model for cell value operations."""

    value: str | int | float | bool | None
    coordinates: dict[str, str | int]
    data_type: str


class SetCellResult(BaseToolResponse):
    """Response model for cell update operations."""

    coordinates: dict[str, str | int]
    old_value: str | int | float | bool | None
    new_value: str | int | float | bool | None
    data_type: str


class RowDataResult(BaseToolResponse):
    """Response model for row data operations."""

    session_id: str
    row_index: int
    data: dict[str, str | int | float | bool | None]
    columns: list[str]


class ColumnDataResult(BaseToolResponse):
    """Response model for column data operations."""

    session_id: str
    column: str
    values: list[str | int | float | bool | None]
    total_values: int
    start_row: int | None = None
    end_row: int | None = None


class InsertRowResult(BaseToolResponse):
    """Response model for row insertion operations."""

    operation: str = "insert_row"
    row_index: int
    rows_before: int
    rows_after: int
    data_inserted: dict[str, str | int | float | bool | None]
    columns: list[str]
    session_id: str


class DeleteRowResult(BaseToolResponse):
    """Response model for row deletion operations."""

    session_id: str
    operation: str = "delete_row"
    row_index: int
    rows_before: int
    rows_after: int
    deleted_data: dict[
        str, CsvCellValue
    ]  # Deleted row data structure varies by dataset but values are typed


class UpdateRowResult(BaseToolResponse):
    """Response model for row update operations."""

    operation: str = "update_row"
    row_index: int
    columns_updated: list[str]
    old_values: dict[str, str | int | float | bool | None]
    new_values: dict[str, str | int | float | bool | None]
    changes_made: int


# =============================================================================
# DATA TOOL RESPONSES
# =============================================================================


class FilterOperationResult(BaseToolResponse):
    """Response model for row filtering operations."""

    session_id: str
    rows_before: int
    rows_after: int
    rows_filtered: int
    conditions_applied: int


class ColumnOperationResult(BaseToolResponse):
    """Response model for column operations (add, remove, rename, etc.)."""

    session_id: str
    operation: str
    rows_affected: int
    columns_affected: list[str]
    original_sample: list[CsvCellValue] | None = (
        None  # Column samples can contain mixed CSV data types
    )
    updated_sample: list[CsvCellValue] | None = (
        None  # Column samples can contain mixed CSV data types
    )
    # Additional fields for specific operations
    part_index: int | None = None
    transform: str | None = None
    nulls_filled: int | None = None


class SortDataResult(BaseToolResponse):
    """Response model for data sorting operations."""

    session_id: str
    sorted_by: list[str]
    ascending: list[bool]


class SelectColumnsResult(BaseToolResponse):
    """Response model for column selection operations."""

    session_id: str
    selected_columns: list[str]
    columns_before: int
    columns_after: int


class RenameColumnsResult(BaseToolResponse):
    """Response model for column rename operations."""

    session_id: str
    renamed: dict[str, str]
    columns: list[str]


# =============================================================================
# TYPE UNIONS FOR FLEXIBILITY
# =============================================================================

# Union type for all possible tool responses
ToolResponse = (
    HealthResult
    | ServerInfoResult
    | StatisticsResult
    | CorrelationResult
    | ValueCountsResult
    | OutliersResult
    | ProfileResult
    | DataSummaryResult
    | ColumnStatisticsResult
    | GroupAggregateResult
    | InspectDataResult
    | FindCellsResult
    | CellValueResult
    | SetCellResult
    | RowDataResult
    | ColumnDataResult
    | InsertRowResult
    | DeleteRowResult
    | UpdateRowResult
    | FilterOperationResult
    | ColumnOperationResult
    | SortDataResult
    | SelectColumnsResult
    | RenameColumnsResult
)
