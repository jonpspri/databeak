"""Unified Pydantic response models for all MCP tools.

This module consolidates all tool response models to eliminate type conflicts and provide consistent
structured responses across DataBeak's MCP interface.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Type alias for CSV cell values - more specific than Any while still flexible
CsvCellValue = str | int | float | bool | None

# =============================================================================
# NESTED PYDANTIC MODELS FOR STRUCTURED DATA
# =============================================================================


class SessionInfo(BaseModel):
    """Session information in list results."""

    session_id: str = Field(description="Unique session identifier")
    created_at: str = Field(description="Session creation timestamp (ISO format)")
    last_accessed: str = Field(description="Last access timestamp (ISO format)")
    row_count: int = Field(description="Number of rows in dataset")
    column_count: int = Field(description="Number of columns in dataset")
    columns: list[str] = Field(description="List of column names")
    memory_usage_mb: float = Field(description="Memory usage in megabytes")
    file_path: str | None = Field(
        default=None, description="Original file path if loaded from file"
    )


class DataTypeInfo(BaseModel):
    """Data type information for columns."""

    type: Literal["int64", "float64", "object", "bool", "datetime64", "category"] = Field(
        description="Pandas data type"
    )
    nullable: bool = Field(description="Whether column allows null values")
    unique_count: int = Field(description="Number of unique values")
    null_count: int = Field(description="Number of null/missing values")


class MissingDataInfo(BaseModel):
    """Missing data summary."""

    total_missing: int = Field(description="Total number of missing values across all columns")
    missing_by_column: dict[str, int] = Field(description="Missing value count for each column")
    missing_percentage: float = Field(description="Percentage of missing values (0-100)")


class DataPreview(BaseModel):
    """Data preview with row samples."""

    rows: list[dict[str, CsvCellValue]] = Field(description="Sample rows from dataset")
    row_count: int = Field(description="Total number of rows in dataset")
    column_count: int = Field(description="Total number of columns in dataset")
    truncated: bool = Field(default=False, description="Whether preview is truncated")


class CellLocation(BaseModel):
    """Cell location and value information."""

    row: int = Field(description="Row index (0-based)")
    column: str = Field(description="Column name")
    value: CsvCellValue = Field(description="Cell value (str, int, float, bool, or None)")


class BaseToolResponse(BaseModel):
    """Base response model for all MCP tool operations."""

    success: bool = Field(default=True, description="Whether operation completed successfully")


# =============================================================================
# SYSTEM TOOL RESPONSES
# =============================================================================


class HealthResult(BaseToolResponse):
    """Response model for system health check."""

    status: str = Field(description="Server health status: healthy, degraded, or unhealthy")
    version: str = Field(description="DataBeak server version")
    active_sessions: int = Field(description="Number of currently active data sessions")
    max_sessions: int = Field(description="Maximum allowed concurrent sessions")
    session_ttl_minutes: int = Field(description="Session timeout in minutes")


class ServerInfoResult(BaseToolResponse):
    """Response model for server information and capabilities."""

    name: str = Field(description="Server name and identification")
    version: str = Field(description="Current server version")
    description: str = Field(description="Server description and purpose")
    capabilities: dict[str, list[str]] = Field(
        description="Available operations organized by category"
    )
    supported_formats: list[str] = Field(description="Supported file formats and extensions")
    max_file_size_mb: int = Field(description="Maximum file size limit in MB")
    session_timeout_minutes: int = Field(description="Default session timeout in minutes")


# =============================================================================
# ROW TOOL RESPONSES
# =============================================================================


class CellValueResult(BaseToolResponse):
    """Response model for cell value operations."""

    value: str | int | float | bool | None = Field(description="Cell value (None if null/missing)")
    coordinates: dict[str, str | int] = Field(
        description="Cell coordinates with row index and column name"
    )
    data_type: str = Field(description="Pandas data type of the column")


class SetCellResult(BaseToolResponse):
    """Response model for cell update operations."""

    coordinates: dict[str, str | int] = Field(
        description="Cell coordinates with row index and column name"
    )
    old_value: str | int | float | bool | None = Field(
        description="Previous cell value before update"
    )
    new_value: str | int | float | bool | None = Field(description="New cell value after update")
    data_type: str = Field(description="Pandas data type of the column")


class RowDataResult(BaseToolResponse):
    """Response model for row data operations."""

    session_id: str = Field(description="Session identifier")
    row_index: int = Field(description="Row index (0-based)")
    data: dict[str, str | int | float | bool | None] = Field(
        description="Row data as column name to value mapping"
    )
    columns: list[str] = Field(description="List of column names included in data")


class ColumnDataResult(BaseToolResponse):
    """Response model for column data operations."""

    session_id: str = Field(description="Session identifier")
    column: str = Field(description="Column name")
    values: list[str | int | float | bool | None] = Field(
        description="Column values in specified range"
    )
    total_values: int = Field(description="Number of values returned")
    start_row: int | None = Field(
        default=None, description="Starting row index used (None if from beginning)"
    )
    end_row: int | None = Field(default=None, description="Ending row index used (None if to end)")


class InsertRowResult(BaseToolResponse):
    """Response model for row insertion operations."""

    operation: str = Field(default="insert_row", description="Operation type identifier")
    row_index: int = Field(description="Index where row was inserted")
    rows_before: int = Field(description="Row count before insertion")
    rows_after: int = Field(description="Row count after insertion")
    data_inserted: dict[str, str | int | float | bool | None] = Field(
        description="Actual data that was inserted"
    )
    columns: list[str] = Field(description="Current column names")
    session_id: str = Field(description="Session identifier")


class DeleteRowResult(BaseToolResponse):
    """Response model for row deletion operations."""

    session_id: str = Field(description="Session identifier")
    operation: str = Field(default="delete_row", description="Operation type identifier")
    row_index: int = Field(description="Index of deleted row")
    rows_before: int = Field(description="Row count before deletion")
    rows_after: int = Field(description="Row count after deletion")
    deleted_data: dict[str, CsvCellValue] = Field(description="Data from the deleted row")


class UpdateRowResult(BaseToolResponse):
    """Response model for row update operations."""

    operation: str = Field(default="update_row", description="Operation type identifier")
    row_index: int = Field(description="Index of updated row")
    columns_updated: list[str] = Field(description="Names of columns that were updated")
    old_values: dict[str, str | int | float | bool | None] = Field(
        description="Previous values for updated columns"
    )
    new_values: dict[str, str | int | float | bool | None] = Field(
        description="New values for updated columns"
    )
    changes_made: int = Field(description="Number of columns that were changed")


# =============================================================================
# DATA TOOL RESPONSES
# =============================================================================


class FilterOperationResult(BaseToolResponse):
    """Response model for row filtering operations."""

    session_id: str = Field(description="Session identifier")
    rows_before: int = Field(description="Row count before filtering")
    rows_after: int = Field(description="Row count after filtering")
    rows_filtered: int = Field(description="Number of rows removed by filter")
    conditions_applied: int = Field(description="Number of filter conditions applied")


class ColumnOperationResult(BaseToolResponse):
    """Response model for column operations (add, remove, rename, etc.)."""

    session_id: str = Field(description="Session identifier")
    operation: str = Field(description="Type of operation performed")
    rows_affected: int = Field(description="Number of rows affected by operation")
    columns_affected: list[str] = Field(description="Names of columns affected")
    original_sample: list[CsvCellValue] | None = Field(
        default=None, description="Sample values before operation"
    )
    updated_sample: list[CsvCellValue] | None = Field(
        default=None, description="Sample values after operation"
    )
    # Additional fields for specific operations
    part_index: int | None = Field(default=None, description="Part index for split operations")
    transform: str | None = Field(default=None, description="Transform description")
    nulls_filled: int | None = Field(default=None, description="Number of null values filled")
    rows_removed: int | None = Field(
        default=None, description="Number of rows removed (for remove_duplicates)"
    )
    values_filled: int | None = Field(
        default=None, description="Number of values filled (for fill_missing_values)"
    )


class SortDataResult(BaseToolResponse):
    """Response model for data sorting operations."""

    session_id: str = Field(description="Session identifier")
    sorted_by: list[str] = Field(description="Column names used for sorting")
    ascending: list[bool] = Field(
        description="Sort direction for each column (True=ascending, False=descending)"
    )
    rows_processed: int = Field(description="Number of rows that were sorted")


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
