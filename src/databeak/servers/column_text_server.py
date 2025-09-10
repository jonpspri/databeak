"""FastMCP server for text and string column operations.

This server provides specialized text manipulation operations for column data.
"""

from __future__ import annotations

from typing import Any, Literal

from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from ..models.tool_responses import ColumnOperationResult
from ..tools.transformations import extract_from_column as _extract_from_column
from ..tools.transformations import fill_column_nulls as _fill_column_nulls
from ..tools.transformations import replace_in_column as _replace_in_column
from ..tools.transformations import split_column as _split_column
from ..tools.transformations import strip_column as _strip_column
from ..tools.transformations import transform_column_case as _transform_column_case

# Type aliases
CellValue = str | int | float | bool | None

# =============================================================================
# PYDANTIC MODELS FOR REQUEST PARAMETERS
# =============================================================================


class RegexPattern(BaseModel):
    """Regex pattern specification."""

    pattern: str = Field(description="Regular expression pattern")
    flags: list[Literal["IGNORECASE", "MULTILINE", "DOTALL"]] = Field(
        default_factory=list, description="Regex flags to apply"
    )


class SplitConfig(BaseModel):
    """Configuration for column splitting."""

    delimiter: str = Field(default=" ", description="String to split on")
    max_splits: int | None = Field(default=None, description="Maximum number of splits")
    expand: bool = Field(default=False, description="Expand into multiple columns")


# =============================================================================
# SERVER INITIALIZATION
# =============================================================================

column_text_server = FastMCP("DataBeak Text Column Operations Server")


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================


@column_text_server.tool
async def replace_in_column(
    session_id: str,
    column: str,
    pattern: str,
    replacement: str,
    regex: bool = True,
    ctx: Context | None = None,
) -> ColumnOperationResult:
    r"""Replace patterns in a column with replacement text.

    Args:
        session_id: Session identifier
        column: Column name to update
        pattern: Pattern to search for (regex or literal string)
        replacement: Replacement string
        regex: Whether to treat pattern as regex (default: True)
        ctx: FastMCP context

    Returns:
        ColumnOperationResult with replacement details

    Examples:
        # Replace with regex
        replace_in_column(session_id, "name", r"Mr\.", "Mister")

        # Remove non-digits from phone numbers
        replace_in_column(session_id, "phone", r"\D", "", regex=True)

        # Simple string replacement
        replace_in_column(session_id, "status", "N/A", "Unknown", regex=False)

        # Replace multiple spaces with single space
        replace_in_column(session_id, "description", r"\s+", " ")
    """
    return await _replace_in_column(session_id, column, pattern, replacement, regex, ctx)


@column_text_server.tool
async def extract_from_column(
    session_id: str,
    column: str,
    pattern: str,
    expand: bool = False,
    ctx: Context | None = None,
) -> ColumnOperationResult:
    r"""Extract patterns from a column using regex with capturing groups.

    Args:
        session_id: Session identifier
        column: Column name to extract from
        pattern: Regex pattern with capturing groups
        expand: Whether to expand multiple groups into separate columns
        ctx: FastMCP context

    Returns:
        ColumnOperationResult with extraction details

    Examples:
        # Extract email parts
        extract_from_column(session_id, "email", r"(.+)@(.+)")

        # Extract code components
        extract_from_column(session_id, "product_code", r"([A-Z]{2})-(\d+)")

        # Extract and expand into multiple columns
        extract_from_column(session_id, "full_name", r"(\w+)\s+(\w+)", expand=True)

        # Extract year from date string
        extract_from_column(session_id, "date", r"\d{4}")
    """
    return await _extract_from_column(session_id, column, pattern, expand, ctx)


@column_text_server.tool
async def split_column(
    session_id: str,
    column: str,
    delimiter: str = " ",
    part_index: int | None = None,
    expand_to_columns: bool = False,
    new_columns: list[str] | None = None,
    ctx: Context | None = None,
) -> ColumnOperationResult:
    """Split column values by delimiter.

    Args:
        session_id: Session identifier
        column: Column name to split
        delimiter: String to split on (default: space)
        part_index: Which part to keep (0-based). None keeps first part
        expand_to_columns: Whether to expand splits into multiple columns
        new_columns: Names for new columns when expanding
        ctx: FastMCP context

    Returns:
        ColumnOperationResult with split details

    Examples:
        # Keep first part of split
        split_column(session_id, "full_name", " ", part_index=0)

        # Keep last part
        split_column(session_id, "email", "@", part_index=1)

        # Expand into multiple columns
        split_column(session_id, "address", ",", expand_to_columns=True)

        # Expand with custom column names
        split_column(session_id, "name", " ", expand_to_columns=True,
                    new_columns=["first_name", "last_name"])
    """
    return await _split_column(
        session_id, column, delimiter, part_index, expand_to_columns, new_columns, ctx
    )


@column_text_server.tool
async def transform_column_case(
    session_id: str,
    column: str,
    transform: Literal["upper", "lower", "title", "capitalize"],
    ctx: Context | None = None,
) -> ColumnOperationResult:
    """Transform the case of text in a column.

    Args:
        session_id: Session identifier
        column: Column name to transform
        transform: Type of case transformation:
            - "upper": Convert to UPPERCASE
            - "lower": Convert to lowercase
            - "title": Convert to Title Case
            - "capitalize": Capitalize first letter only
        ctx: FastMCP context

    Returns:
        ColumnOperationResult with transformation details

    Examples:
        # Convert to uppercase
        transform_column_case(session_id, "code", "upper")

        # Convert names to title case
        transform_column_case(session_id, "name", "title")

        # Convert to lowercase for comparison
        transform_column_case(session_id, "email", "lower")

        # Capitalize sentences
        transform_column_case(session_id, "description", "capitalize")
    """
    return await _transform_column_case(session_id, column, transform, ctx)


@column_text_server.tool
async def strip_column(
    session_id: str,
    column: str,
    chars: str | None = None,
    ctx: Context | None = None,
) -> ColumnOperationResult:
    """Strip whitespace or specified characters from column values.

    Args:
        session_id: Session identifier
        column: Column name to strip
        chars: Characters to strip (None for whitespace)
        ctx: FastMCP context

    Returns:
        ColumnOperationResult with strip details

    Examples:
        # Remove leading/trailing whitespace
        strip_column(session_id, "name")

        # Remove specific characters
        strip_column(session_id, "phone", "()")

        # Clean currency values
        strip_column(session_id, "price", "$,")

        # Remove quotes
        strip_column(session_id, "quoted_text", "'\"")
    """
    return await _strip_column(session_id, column, chars, ctx)


@column_text_server.tool
async def fill_column_nulls(
    session_id: str, column: str, value: Any, ctx: Context | None = None
) -> ColumnOperationResult:
    """Fill null/NaN values in a specific column with a specified value.

    Args:
        session_id: Session identifier
        column: Column name to fill
        value: Value to use for filling nulls
        ctx: FastMCP context

    Returns:
        ColumnOperationResult with fill details

    Examples:
        # Fill missing names with "Unknown"
        fill_column_nulls(session_id, "name", "Unknown")

        # Fill missing ages with 0
        fill_column_nulls(session_id, "age", 0)

        # Fill missing status with default
        fill_column_nulls(session_id, "status", "pending")

        # Fill missing scores with -1
        fill_column_nulls(session_id, "score", -1)
    """
    return await _fill_column_nulls(session_id, column, value, ctx)
