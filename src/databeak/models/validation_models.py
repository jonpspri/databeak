"""Shared validation models for DataBeak MCP tools.

This module provides common Pydantic models and validators to standardize validation patterns across
all MCP tools, addressing Issue #77.
"""

from __future__ import annotations

import re

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import get_session_manager
from .csv_session import CSVSession


class RowIndexField(BaseModel):
    """Reusable field for row index validation with bounds checking."""

    model_config = ConfigDict(extra="forbid")

    row_index: int = Field(ge=0, description="Row index (0-based, non-negative)")

    @field_validator("row_index")
    @classmethod
    def validate_row_index(cls, v: int) -> int:
        """Validate row index is non-negative."""
        if v < 0:
            raise ValueError("Row index must be non-negative")
        return v


class ColumnNameField(BaseModel):
    """Reusable field for column name validation."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(min_length=1, description="Column name")

    @field_validator("column")
    @classmethod
    def validate_column_name(cls, v: str) -> str:
        """Validate column name is not empty."""
        if not v.strip():
            raise ValueError("Column name cannot be empty")
        return v.strip()


class MultipleColumnsField(BaseModel):
    """Reusable field for multiple column names validation."""

    model_config = ConfigDict(extra="forbid")

    columns: list[str] = Field(min_length=1, description="List of column names")

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v: list[str]) -> list[str]:
        """Validate column names are not empty."""
        if not v:
            raise ValueError("At least one column must be specified")

        validated_columns = []
        for col in v:
            if not isinstance(col, str) or not col.strip():
                raise ValueError(f"Invalid column name: {col}")
            validated_columns.append(col.strip())

        return validated_columns


class RegexPatternField(BaseModel):
    """Reusable field for regex pattern validation."""

    model_config = ConfigDict(extra="forbid")

    pattern: str | None = Field(default=None, description="Regular expression pattern")

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str | None) -> str | None:
        """Validate that pattern is a valid regular expression."""
        if v is None:
            return v

        try:
            re.compile(v)
            return v
        except re.error as e:
            raise ValueError(f"Invalid regular expression: {e}") from e


# Runtime validation functions for DataFrame-dependent checks
def validate_session_and_data(ctx: Context) -> tuple[str, CSVSession]:
    """Validate session exists and has data loaded.

    This replaces the common pattern:
    if not session or session.df is None:
        raise ToolError("Invalid session or no data loaded")

    Args:
        ctx: FastMCP context for session access

    Returns:
        Tuple of (session_id, session)

    Raises:
        ToolError: If session is invalid or no data is loaded
    """
    session_id = ctx.session_id
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if not session or session.df is None:
        raise ToolError("Invalid session or no data loaded")

    # session is guaranteed to be non-None and have df at this point
    return session_id, session


def validate_row_bounds(df_length: int, row_index: int) -> None:
    """Validate row index is within DataFrame bounds.

    Args:
        df_length: Length of the DataFrame
        row_index: Row index to validate

    Raises:
        ToolError: If row index is out of bounds
    """
    if row_index < 0 or row_index >= df_length:
        raise ToolError(f"Row index {row_index} out of range (0-{df_length - 1})")


def validate_column_exists(df_columns: list[str], column: str) -> None:
    """Validate column exists in DataFrame.

    Args:
        df_columns: List of DataFrame column names
        column: Column name to validate

    Raises:
        ColumnNotFoundError: If column does not exist
    """
    if column not in df_columns:
        from ..exceptions import ColumnNotFoundError

        raise ColumnNotFoundError(column, df_columns)


def validate_columns_exist(df_columns: list[str], columns: list[str]) -> None:
    """Validate multiple columns exist in DataFrame.

    Args:
        df_columns: List of DataFrame column names
        columns: Column names to validate

    Raises:
        ColumnNotFoundError: If any column does not exist
    """
    missing_columns = [col for col in columns if col not in df_columns]
    if missing_columns:
        from ..exceptions import ColumnNotFoundError

        # Use first missing column for backward compatibility
        raise ColumnNotFoundError(missing_columns[0], df_columns)


# Compound validation models for common parameter combinations
class RowColumnRequest(RowIndexField, ColumnNameField):
    """Common request model for operations requiring both row index and column name."""

    model_config = ConfigDict(extra="forbid")


class MultipleRowsRequest(BaseModel):
    """Request model for operations on multiple rows."""

    model_config = ConfigDict(extra="forbid")

    row_indices: list[int] = Field(description="List of row indices (0-based)")

    @field_validator("row_indices")
    @classmethod
    def validate_row_indices(cls, v: list[int]) -> list[int]:
        """Validate all row indices are non-negative."""
        if not v:
            raise ValueError("At least one row index must be specified")

        for idx in v:
            if idx < 0:
                raise ValueError(f"Row index {idx} must be non-negative")

        return v


class RangeRequest(BaseModel):
    """Request model for range-based operations."""

    model_config = ConfigDict(extra="forbid")

    start_row: int | None = Field(None, ge=0, description="Starting row index (inclusive)")
    end_row: int | None = Field(None, ge=0, description="Ending row index (exclusive)")

    @field_validator("start_row", "end_row")
    @classmethod
    def validate_row_indices(cls, v: int | None) -> int | None:
        """Validate row indices are non-negative."""
        if v is not None and v < 0:
            raise ValueError("Row indices must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_range(self) -> RangeRequest:
        """Validate end_row is greater than start_row."""
        if (
            self.end_row is not None
            and self.start_row is not None
            and self.end_row <= self.start_row
        ):
            raise ValueError("end_row must be greater than start_row")
        return self
