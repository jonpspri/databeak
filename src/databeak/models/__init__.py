"""Data models for CSV Editor MCP Server."""

from __future__ import annotations

# Type alias defined here to avoid circular imports
CellValue = str | int | float | bool | None

from .data_models import (  # noqa: E402
    AggregateFunction,
    ColumnSchema,
    ComparisonOperator,
    DataPreview,
    DataQualityRule,
    DataSchema,
    DataStatistics,
    DataType,
    ExportFormat,
    FilterCondition,
    LogicalOperator,
    OperationResult,
    OperationType,  # Still used in some places, can be removed in cleanup phase
    SessionInfo,
    SortSpec,
)
from .session_service import (  # noqa: E402
    SessionManagerProtocol,
    SessionService,
    SessionServiceFactory,
    get_default_session_service_factory,
)

__all__ = [
    "AggregateFunction",
    "CellValue",
    "ColumnSchema",
    "ComparisonOperator",
    "DataPreview",
    "DataQualityRule",
    "DataSchema",
    "DataStatistics",
    "DataType",
    "ExportFormat",
    "FilterCondition",
    "LogicalOperator",
    "OperationResult",
    "OperationType",
    "SessionInfo",
    "SessionManagerProtocol",
    "SessionService",
    "SessionServiceFactory",
    "SortSpec",
    "get_default_session_service_factory",
]
