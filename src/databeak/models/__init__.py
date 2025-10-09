"""Data models for CSV Editor MCP Server."""

from __future__ import annotations

# Type alias defined here to avoid circular imports
CellValue = str | int | float | bool | None

from .data_models import (  # noqa: E402
    AggregateFunction,
    ColumnSchema,
    ComparisonOperator,
    DataPreview,
    DataStatistics,
    DataType,
    FilterCondition,
    LogicalOperator,
    OperationResult,
    SessionInfo,
    SortSpec,
)

__all__ = [
    "AggregateFunction",
    "CellValue",
    "ColumnSchema",
    "ComparisonOperator",
    "DataPreview",
    "DataStatistics",
    "DataType",
    "FilterCondition",
    "LogicalOperator",
    "OperationResult",
    "SessionInfo",
    "SortSpec",
]
