"""TypedDict definitions for DataBeak data structures.

This module provides specific typed dictionary definitions to replace
generic dict[str, Any] usage throughout the DataBeak codebase, improving
type safety and IDE support.

Author: DataBeak Type Safety Team
Issue: #45 - Reduce Any usage by 70%
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

# Type aliases for common data types
CellValue = str | int | float | bool | None


# Validation and Quality Check Results
class ValidationResult(TypedDict):
    """Result of DataFrame schema validation."""
    valid: bool
    errors: list[str]
    warnings: list[str]


class DataValidationIssues(TypedDict):
    """Issues found during DataFrame validation."""
    errors: list[str]
    warnings: list[str]
    info: dict[str, Any]  # type: ignore[explicit-any]  # Flexible validation metadata


class QualityCheckResult(TypedDict):
    """Result of data quality assessment."""
    rule_name: str
    passed: bool
    score: float
    message: str
    details: NotRequired[dict[str, Any]]  # Flexible for rule-specific data


class DataStatistics(TypedDict):
    """Statistical summary of column data."""
    count: int
    mean: NotRequired[float]  # Only for numeric columns
    std: NotRequired[float]   # Only for numeric columns
    min: NotRequired[CellValue]
    max: NotRequired[CellValue]
    unique_count: int
    null_count: int
    dtype: str


class ColumnProfile(TypedDict):
    """Comprehensive column profiling information."""
    name: str
    dtype: str
    statistics: DataStatistics
    sample_values: list[CellValue]
    quality_issues: list[str]


# Session and Operation Metadata
class SessionMetadata(TypedDict):
    """Session state and configuration metadata."""
    created_at: str
    last_accessed: str
    auto_save_enabled: bool
    operations_count: int
    data_shape: NotRequired[tuple[int, int]]  # (rows, columns) if data loaded


class OperationMetadata(TypedDict):
    """Metadata for tracking operations in session history."""
    operation_type: str
    timestamp: str
    parameters: dict[str, CellValue]
    rows_affected: NotRequired[int]
    columns_affected: NotRequired[list[str]]
    execution_time_ms: NotRequired[float]


class FilterConditionDict(TypedDict):
    """Filter condition as dictionary (for legacy compatibility)."""
    column: str
    operator: str
    value: CellValue
    case_sensitive: NotRequired[bool]


class SortSpecification(TypedDict):
    """Sort specification for column sorting."""
    column: str
    ascending: bool


# I/O and Data Processing
class CsvReadParams(TypedDict):
    """Parameters for CSV reading operations."""
    sep: NotRequired[str]
    header: NotRequired[int | None]
    names: NotRequired[list[str]]
    dtype: NotRequired[dict[str, str]]
    parse_dates: NotRequired[list[str]]
    encoding: NotRequired[str]
    skiprows: NotRequired[int]
    nrows: NotRequired[int]


class ExportOptions(TypedDict):
    """Options for data export operations."""
    format: str  # 'csv', 'json', 'excel', etc.
    include_index: bool
    encoding: NotRequired[str]
    sep: NotRequired[str]  # For CSV
    sheet_name: NotRequired[str]  # For Excel


# Data Transformation Structures
class TransformationStep(TypedDict):
    """Single step in a data transformation pipeline."""
    operation: str
    parameters: dict[str, CellValue]
    target_columns: NotRequired[list[str]]


class TransformationPipeline(TypedDict):
    """Complete transformation pipeline specification."""
    steps: list[TransformationStep]
    description: NotRequired[str]
    validation_rules: NotRequired[list[str]]


# Internal operation results (for legacy transformation functions)
class ColumnSelectionResult(TypedDict):
    """Result of internal column selection operation."""
    session_id: str
    selected_columns: list[str]
    columns_before: int
    columns_after: int


class RowUpdateResult(TypedDict):
    """Result of internal row update operation."""
    session_id: str
    row_index: int
    updated_fields: dict[str, CellValue]
    columns_modified: list[str]


class ColumnRenameResult(TypedDict):
    """Result of internal column rename operation."""
    session_id: str
    renamed: dict[str, str]  # old_name -> new_name mapping
    columns: list[str]  # Final column list after rename


# Tool Response Components
class OperationResult(TypedDict):
    """Standard operation result structure."""
    success: bool
    operation_type: str
    rows_affected: int
    columns_affected: list[str]
    execution_time_ms: float
    message: NotRequired[str]


class ErrorDetails(TypedDict):
    """Detailed error information."""
    error_type: str
    message: str
    parameter: NotRequired[str]
    suggested_fix: NotRequired[str]


# Discovery and Analysis Results
class ColumnAnalysis(TypedDict):
    """Analysis results for a single column."""
    column_name: str
    data_type: str
    unique_values: int
    null_percentage: float
    sample_values: list[CellValue]
    patterns: NotRequired[list[str]]
    anomalies: NotRequired[list[str]]


class DataProfileResult(TypedDict):
    """Complete data profiling results."""
    total_rows: int
    total_columns: int
    memory_usage_mb: float
    column_analyses: list[ColumnAnalysis]
    correlations: NotRequired[dict[str, dict[str, float]]]
    summary_statistics: NotRequired[dict[str, DataStatistics]]


# Configuration and Settings
class ServerConfig(TypedDict):
    """Server configuration parameters."""
    host: str
    port: int
    debug: bool
    auto_save: bool
    session_timeout_minutes: int
    max_memory_mb: NotRequired[int]


class ToolConfig(TypedDict):
    """Individual tool configuration."""
    enabled: bool
    timeout_seconds: NotRequired[int]
    memory_limit_mb: NotRequired[int]
    validation_level: NotRequired[str]  # 'strict', 'normal', 'permissive'


# Data Preview Structures
class DataPreviewRecord(TypedDict):
    """Single record in data preview with row index."""
    __row_index__: int  # Original DataFrame row index
    # Additional fields are column data as CellValue


class DataPreviewResult(TypedDict):
    """Complete data preview with metadata."""
    records: list[dict[str, CellValue]]  # Preview records with actual column data
    total_rows: int
    total_columns: int  # Required by io_server.py
    columns: list[str]
    preview_rows: int


class InternalDataSummary(TypedDict):
    """Internal data summary structure (not an MCP tool response)."""
    session_id: str
    shape: tuple[int, int]  # (rows, columns)
    columns: list[str]
    dtypes: dict[str, str]
    memory_usage_mb: float
    null_counts: dict[str, int]
    preview: DataPreviewResult


# Legacy compatibility - gradually replace these
DataDict = dict[str, CellValue]  # Structured data with known value types
MetadataDict = dict[str, str | int | float | bool]  # Metadata with primitive types
ConfigDict = dict[str, str | int | bool]  # Configuration with known types
