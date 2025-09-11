"""Unified Pydantic response models for all MCP tools.

This module consolidates all tool response models to eliminate type conflicts and provide consistent
structured responses across DataBeak's MCP interface.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

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


class CellLocation(BaseModel):
    """Cell location and value information."""

    row: int
    column: str
    value: CsvCellValue  # CSV cells can contain str, int, float, bool, or None


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
    rows_removed: int | None = None  # For remove_duplicates operation
    values_filled: int | None = None  # For fill_missing_values operation


class SortDataResult(BaseToolResponse):
    """Response model for data sorting operations."""

    session_id: str
    sorted_by: list[str]
    ascending: list[bool]
    rows_processed: int


# =============================================================================
# TYPE UNIONS FOR FLEXIBILITY
# =============================================================================

# Union type for all possible tool responses
ToolResponse = (
    HealthResult
    | ServerInfoResult
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
)
