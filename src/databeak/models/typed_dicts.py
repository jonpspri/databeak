"""TypedDict definitions for DataBeak data structures.

This module provides specific typed dictionary definitions to replace
generic dict[str, Any] usage throughout the DataBeak codebase, improving
type safety and IDE support.

Author: DataBeak Type Safety Team
Issue: #45 - Reduce Any usage by 70%
"""

from __future__ import annotations

from typing import Any, TypedDict

from databeak.models import CellValue

__all__ = [
    "CellValue",
    "DataDict",
    "DataPreviewResult",
    "DataValidationIssues",
    "InternalDataSummary",
    "MetadataDict",
    "ValidationResult",
]


# Validation Results (used in validators and data models)
class ValidationResult(TypedDict):
    """Result of DataFrame schema validation."""

    valid: bool
    errors: list[str]
    warnings: list[str]


class DataValidationIssues(TypedDict):
    """Issues found during DataFrame validation."""

    errors: list[str]
    warnings: list[str]
    info: dict[str, Any]  # Any justified: flexible validation metadata


# Data Preview Structure (used in services for internal operations)
class DataPreviewResult(TypedDict):
    """Complete data preview with metadata.

    Used internally by data_operations.py. Fields map to DataPreview Pydantic model
    but with different naming for backward compatibility.
    """

    records: list[dict[str, CellValue]]  # Preview records with actual column data
    total_rows: int
    total_columns: int
    columns: list[str]
    preview_rows: int


# Internal data structures (used in services)
class InternalDataSummary(TypedDict):
    """Internal data summary structure (not an MCP tool response)."""

    session_id: str
    shape: tuple[int, int]  # (rows, columns)
    columns: list[str]
    dtypes: dict[str, str]
    memory_usage_mb: float
    null_counts: dict[str, int]
    preview: DataPreviewResult


# Type aliases for common data patterns
DataDict = dict[str, CellValue]  # Structured data with known value types
MetadataDict = dict[str, str | int | float | bool]  # Metadata with primitive types
