"""Core data models and enums for DataBeak operations."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from . import CellValue

# Type aliases for common data types
type FilterValue = CellValue | list[CellValue]

if TYPE_CHECKING:
    pass


class DataType(str, Enum):
    """Supported data types for columns."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    MIXED = "mixed"


class ComparisonOperator(str, Enum):
    """Comparison operators for filtering."""

    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUALS = ">="
    LESS_THAN_OR_EQUALS = "<="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class LogicalOperator(str, Enum):
    """Logical operators for combining conditions."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class AggregateFunction(str, Enum):
    """Aggregate functions for data analysis."""

    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    STD = "std"
    VAR = "var"
    FIRST = "first"
    LAST = "last"


class FilterCondition(BaseModel):
    """A single filter condition."""

    column: str = Field(..., description="Column name to filter on")
    operator: ComparisonOperator = Field(..., description="Comparison operator")
    value: FilterValue = Field(default=None, description="Value to compare against")

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, v: FilterValue, info: Any) -> FilterValue:
        """Validate value based on operator."""
        operator = info.data.get("operator") if hasattr(info, "data") else None
        if operator in [ComparisonOperator.IS_NULL, ComparisonOperator.IS_NOT_NULL]:
            return None
        if operator in [
            ComparisonOperator.IN,
            ComparisonOperator.NOT_IN,
        ] and not isinstance(v, list):
            return [v]
        return v


class SortSpec(BaseModel):
    """Specification for sorting data."""

    column: str = Field(..., description="Column to sort by")
    ascending: bool = Field(default=True, description="Sort in ascending order")


class ColumnSchema(BaseModel):
    """Schema definition for a column."""

    name: str = Field(..., description="Column name")
    dtype: DataType = Field(..., description="Data type")
    nullable: bool = Field(default=True, description="Whether column can contain null values")
    unique: bool = Field(default=False, description="Whether values must be unique")
    min_value: float | int | str | None = Field(default=None, description="Minimum value")
    max_value: float | int | str | None = Field(default=None, description="Maximum value")
    allowed_values: list[CellValue] | None = Field(
        default=None,
        description="List of allowed values",
    )
    pattern: str | None = Field(default=None, description="Regex pattern for validation")


class OperationResult(BaseModel):
    """Result of a data operation."""

    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field(..., description="Result message")
    rows_affected: int | None = Field(default=None, description="Number of rows affected")
    columns_affected: list[str] | None = Field(default=None, description="Columns affected")
    data: dict[str, Any] | None = Field(default=None, description="Additional result data")
    error: str | None = Field(default=None, description="Error message if failed")
    warnings: list[str] | None = Field(default=None, description="Warning messages")


class SessionInfo(BaseModel):
    """Information about a data session."""

    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(..., description="Session creation time")
    last_accessed: datetime = Field(..., description="Last access time")
    row_count: int = Field(..., description="Number of rows in dataset")
    column_count: int = Field(..., description="Number of columns")
    columns: list[str] = Field(..., description="Column names")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    operations_count: int = Field(default=0, description="Number of operations performed")
    file_path: str | None = Field(default=None, description="Source file path")


class DataStatistics(BaseModel):
    """Statistical summary of data."""

    column: str = Field(..., description="Column name")
    dtype: str = Field(..., description="Data type")
    count: int = Field(..., description="Non-null count")
    null_count: int = Field(..., description="Null count")
    unique_count: int = Field(..., description="Unique value count")
    mean: float | None = Field(default=None, description="Mean (numeric only)")
    std: float | None = Field(default=None, description="Standard deviation (numeric only)")
    min: CellValue = Field(default=None, description="Minimum value")
    max: CellValue = Field(default=None, description="Maximum value")
    q25: float | None = Field(default=None, description="25th percentile (numeric only)")
    q50: float | None = Field(default=None, description="50th percentile (numeric only)")
    q75: float | None = Field(default=None, description="75th percentile (numeric only)")


class DataPreview(BaseModel):
    """Data preview with row samples."""

    rows: list[dict[str, CellValue]] = Field(description="Sample rows from dataset")
    row_count: int = Field(description="Total number of rows in dataset")
    column_count: int = Field(description="Total number of columns in dataset")
    truncated: bool = Field(default=False, description="Whether preview is truncated")
    # top_values: dict[str, int] | None = Field(
    #     default=None, description="Top 10 most frequent values"
    # )
