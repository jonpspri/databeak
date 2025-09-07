"""Unified Pydantic response models for all MCP tools.

This module consolidates all tool response models to eliminate type conflicts and provide consistent
structured responses across DataBeak's MCP interface.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


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


class ServerInfoResult(BaseModel):
    """Response model for server information and capabilities."""

    name: str
    version: str
    description: str
    capabilities: dict[str, list[str]]
    supported_formats: list[str]
    max_file_size_mb: int
    session_timeout_minutes: int


# =============================================================================
# IO TOOL RESPONSES
# =============================================================================


class LoadResult(BaseToolResponse):
    """Response model for data loading operations."""

    session_id: str
    rows_affected: int
    columns_affected: list[str]
    data: dict[str, Any] | None = None
    memory_usage_mb: float | None = None


class ExportResult(BaseToolResponse):
    """Response model for data export operations."""

    session_id: str
    file_path: str
    format: str
    rows_exported: int
    file_size_mb: float | None = None


class SessionInfoResult(BaseToolResponse):
    """Response model for session information."""

    session_id: str
    created_at: str
    last_modified: str
    data_loaded: bool
    row_count: int | None = None
    column_count: int | None = None
    auto_save_enabled: bool


class SessionListResult(BaseToolResponse):
    """Response model for listing all sessions."""

    sessions: list[dict[str, Any]]
    total_sessions: int
    active_sessions: int


class CloseSessionResult(BaseToolResponse):
    """Response model for session closure operations."""

    session_id: str
    message: str
    data_preserved: bool


# =============================================================================
# ANALYTICS TOOL RESPONSES
# =============================================================================


class StatisticsResult(BaseToolResponse):
    """Response model for statistical analysis operations."""

    session_id: str
    statistics: dict[str, dict[str, float]]
    column_count: int
    numeric_columns: list[str]
    total_rows: int


class CorrelationResult(BaseToolResponse):
    """Response model for correlation matrix operations."""

    session_id: str
    correlation_matrix: dict[str, dict[str, float]]
    method: str
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
    outliers_by_column: dict[str, list[dict[str, Any]]]
    method: str
    threshold: float


class ProfileResult(BaseToolResponse):
    """Response model for comprehensive data profiling."""

    session_id: str
    profile: dict[str, Any]
    total_rows: int
    total_columns: int
    memory_usage_mb: float


class DataSummaryResult(BaseToolResponse):
    """Response model for comprehensive data summary."""

    session_id: str
    coordinate_system: dict[str, str]
    shape: dict[str, int]
    columns: dict[str, Any]
    data_types: dict[str, list[str]]
    missing_data: dict[str, Any]
    memory_usage_mb: float
    preview: dict[str, Any]


class ColumnStatisticsResult(BaseToolResponse):
    """Response model for column-specific statistics."""

    session_id: str
    column: str
    statistics: dict[str, float]
    data_type: str
    non_null_count: int


class GroupAggregateResult(BaseToolResponse):
    """Response model for group-by aggregation operations."""

    session_id: str
    groups: dict[str, dict[str, Any]]
    group_columns: list[str]
    aggregation_functions: dict[str, str | list[str]]
    total_groups: int


class InspectDataResult(BaseToolResponse):
    """Response model for data inspection around specific coordinates."""

    session_id: str
    center_coordinates: dict[str, str | int]
    surrounding_data: dict[str, Any]
    radius: int


class FindCellsResult(BaseToolResponse):
    """Response model for cell search operations."""

    session_id: str
    search_value: str | int | float | bool | None
    matches_found: int
    coordinates: list[dict[str, str | int]]
    search_column: str | None


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
    row_index: int
    rows_before: int
    rows_after: int


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


# =============================================================================
# ERROR RESPONSES
# =============================================================================


class ErrorResult(BaseModel):
    """Standardized error response format."""

    success: bool = False
    error: str
    session_id: str | None = None


# =============================================================================
# TYPE UNIONS FOR FLEXIBILITY
# =============================================================================

# Union type for all possible tool responses
ToolResponse = (
    HealthResult
    | ServerInfoResult
    | LoadResult
    | ExportResult
    | SessionInfoResult
    | SessionListResult
    | CloseSessionResult
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
    | ErrorResult
)
