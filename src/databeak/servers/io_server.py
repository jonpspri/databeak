"""Standalone I/O server for DataBeak using FastMCP server composition.

This module provides a complete I/O server implementation following DataBeak's modular server
architecture pattern. It includes comprehensive CSV loading from web sources and session
management capabilities with robust error handling and AI-optimized documentation.
"""

from __future__ import annotations

import logging
import socket
from abc import ABC, abstractmethod
from io import StringIO
from typing import Annotated, Literal
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Discriminator, Field, NonNegativeInt

from databeak.core.session import get_session_manager, get_session_only

# Import session management and data models from the main package
from databeak.models import DataPreview
from databeak.models.tool_responses import BaseToolResponse
from databeak.services.data_operations import create_data_preview_with_indices
from databeak.utils.validators import validate_url

logger = logging.getLogger(__name__)


# Header configuration types with discriminated union
class HeaderConfig(BaseModel, ABC):
    """Abstract base class for header configuration."""

    mode: str = Field(description="Header detection mode")

    @abstractmethod
    def get_pandas_param(self) -> int | None | Literal["infer"]:
        """Convert to pandas read_csv header parameter."""
        ...


class AutoDetectHeader(HeaderConfig):
    """Auto-detect whether file has headers using pandas inference."""

    mode: Literal["auto"] = "auto"

    def get_pandas_param(self) -> Literal["infer"]:
        """Return pandas parameter for auto-detection."""
        return "infer"


class NoHeader(HeaderConfig):
    """File has no headers - generate default column names (Column_0, Column_1, etc.)."""

    mode: Literal["none"] = "none"

    def get_pandas_param(self) -> None:
        """Return pandas parameter for no headers."""
        return None


class ExplicitHeaderRow(HeaderConfig):
    """Use specific row number as header."""

    mode: Literal["row"] = "row"
    row_number: NonNegativeInt = Field(description="Row number to use as header (0-based)")

    def get_pandas_param(self) -> int:
        """Return pandas parameter for explicit header row."""
        return self.row_number


# Discriminated union type
HeaderConfigUnion = Annotated[
    AutoDetectHeader | NoHeader | ExplicitHeaderRow,
    Discriminator("mode"),
]


def resolve_header_param(config: HeaderConfig) -> int | None | Literal["infer"]:
    """Convert HeaderConfig to pandas read_csv header parameter.

    Args:
        config: Header configuration object

    Returns:
        Value for pandas read_csv header parameter

    """
    return config.get_pandas_param()


# Configuration constants
MAX_MEMORY_USAGE_MB = 1000  # Maximum memory usage in MB for DataFrames
MAX_ROWS = 1_000_000  # Maximum number of rows to prevent memory issues
URL_TIMEOUT_SECONDS = 30  # Timeout for URL downloads
MAX_URL_SIZE_MB = 100  # Maximum download size for URLs

# ============================================================================
# PYDANTIC MODELS FOR I/O OPERATIONS
# ============================================================================


class LoadResult(BaseToolResponse):
    """Response model for data loading operations."""

    rows_affected: int = Field(description="Number of rows loaded")
    columns_affected: list[str] = Field(description="List of column names detected")
    data: DataPreview | None = Field(None, description="Sample of loaded data")
    memory_usage_mb: float | None = Field(None, description="Memory usage in megabytes")


class SessionInfoResult(BaseToolResponse):
    """Response model for session information."""

    created_at: str = Field(description="Creation timestamp (ISO format)")
    last_modified: str = Field(description="Last modification timestamp (ISO format)")
    data_loaded: bool = Field(description="Whether session has data loaded")
    row_count: int | None = Field(None, description="Number of rows if data loaded")
    column_count: int | None = Field(None, description="Number of columns if data loaded")


# ============================================================================
# ENCODING DETECTION UTILITIES
# ============================================================================


# Implementation: prioritizes encoding groups by primary encoding type
# UTF variants -> Windows encodings -> Latin variants -> Asian encodings
# Removes duplicates while preserving priority order
def get_encoding_fallbacks(primary_encoding: str) -> list[str]:
    """Get optimized encoding fallback list based on primary encoding."""
    # Common encoding groups in order of likelihood
    utf_encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-32"]
    windows_encodings = ["cp1252", "windows-1252", "cp1251", "windows-1251"]
    latin_encodings = ["latin1", "iso-8859-1", "iso-8859-15"]
    asian_encodings = ["cp932", "gb2312", "big5", "euc-jp", "euc-kr"]

    # Start with primary encoding
    fallbacks = [primary_encoding] if primary_encoding not in ["utf-8"] else []

    # Add encoding groups based on what's likely to work
    if primary_encoding.startswith("utf"):
        fallbacks.extend([enc for enc in utf_encodings if enc != primary_encoding])
        fallbacks.extend(windows_encodings)
        fallbacks.extend(latin_encodings)
    elif primary_encoding.startswith("cp") or "windows" in primary_encoding:
        fallbacks.extend([enc for enc in windows_encodings if enc != primary_encoding])
        fallbacks.extend(latin_encodings)
        fallbacks.extend([enc for enc in utf_encodings if enc != primary_encoding])
    else:
        # For other encodings, try most common first
        fallbacks.extend(["utf-8", "cp1252", "latin1"])
        fallbacks.extend(windows_encodings)
        fallbacks.extend(asian_encodings)

    # Remove duplicates while preserving order
    seen: set[str] = set()
    result = []
    for enc in fallbacks:
        if enc not in seen:
            seen.add(enc)
            result.append(enc)
    return result


# ============================================================================
# I/O OPERATIONS LOGIC
# ============================================================================


def validate_dataframe_size(df: pd.DataFrame) -> None:
    """Validate DataFrame size against memory and row limits.

    Args:
        df: DataFrame to validate

    Raises:
        ToolError: If DataFrame exceeds size limits

    """
    if len(df) > MAX_ROWS:
        msg = f"File too large: {len(df):,} rows exceeds limit of {MAX_ROWS:,} rows"
        raise ToolError(msg)

    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    if memory_usage_mb > MAX_MEMORY_USAGE_MB:
        msg = f"File too large: {memory_usage_mb:.1f} MB exceeds memory limit of {MAX_MEMORY_USAGE_MB} MB"
        raise ToolError(msg)


# Implementation: HTTP/HTTPS download with security validation and timeouts
# Blocks private networks, validates content-type, enforces size limits
# Uses same encoding fallback strategy as file loading
# Timeout: URL_TIMEOUT_SECONDS, Max download: MAX_URL_SIZE_MB
async def load_csv_from_url(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    url: Annotated[str, Field(description="URL of the CSV file to download and load")],
    encoding: Annotated[
        str, Field(description="Text encoding for file reading (utf-8, latin1, cp1252, etc.)")
    ] = "utf-8",
    delimiter: Annotated[
        str, Field(description="Column delimiter character (comma, tab, semicolon, pipe)")
    ] = ",",
    header_config: Annotated[
        HeaderConfigUnion | None,
        Field(default=None, description="Header detection configuration"),
    ] = None,
) -> LoadResult:
    """Load CSV file from URL into DataBeak session.

    Downloads and parses CSV data with security validation. Returns session ID and data preview for
    further operations.
    """
    # Get session_id from FastMCP context
    session_id = ctx.session_id

    # Handle default header configuration
    if header_config is None:
        header_config = AutoDetectHeader()

    # Validate URL
    is_valid, validated_url = validate_url(url)
    if not is_valid:
        msg = f"Invalid URL: {validated_url}"

        raise ToolError(msg)

    await ctx.info(f"Loading CSV from URL: {url}")
    await ctx.report_progress(0.1)

    # Download with timeout and content-type verification
    try:
        # Pre-download validation with timeout and content-type checking
        await ctx.info("Verifying URL and downloading content...")

        # Set socket timeout for all operations
        socket.setdefaulttimeout(URL_TIMEOUT_SECONDS)

        with urlopen(url, timeout=URL_TIMEOUT_SECONDS) as response:  # nosec B310  # noqa: S310, ASYNC210
            # Verify content-type
            content_type = response.headers.get("Content-Type", "").lower()
            content_length = response.headers.get("Content-Length")

            # Check content type
            valid_content_types = [
                "text/csv",
                "text/plain",
                "application/csv",
                "application/octet-stream",  # Some servers use generic type
                "text/tab-separated-values",
            ]

            if content_type and not any(ct in content_type for ct in valid_content_types):
                logger.warning("Unexpected content-type: %s. Proceeding anyway.", content_type)
                await ctx.info(f"Warning: Content-type is {content_type}, expected CSV format")

            # Check content length
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > MAX_URL_SIZE_MB:
                    msg = f"Download too large: {size_mb:.1f} MB exceeds limit of {MAX_URL_SIZE_MB} MB"

                    raise ToolError(msg)

            await ctx.info(f"Download validated. Content-type: {content_type or 'unknown'}")
            await ctx.report_progress(0.3)

        # Download and parse CSV using pandas with timeout
        df = pd.read_csv(
            url,
            encoding=encoding,
            delimiter=delimiter,
            header=resolve_header_param(header_config),
        )
        validate_dataframe_size(df)

    except (TimeoutError, URLError, HTTPError) as e:
        logger.exception("Network error downloading URL")
        await ctx.error(f"Network error: {e}")
        msg = f"Network error: {e}"

        raise ToolError(msg) from e
    except UnicodeDecodeError as e:
        # Use optimized encoding fallbacks for URL downloads
        df = None
        last_error = e

        await ctx.info("URL encoding error, trying optimized fallbacks...")

        # Use the same optimized fallback strategy
        fallback_encodings = get_encoding_fallbacks(encoding)

        for alt_encoding in fallback_encodings:
            if alt_encoding != encoding:  # Skip the original encoding we already tried
                try:
                    df = pd.read_csv(
                        url,
                        encoding=alt_encoding,
                        delimiter=delimiter,
                        header=resolve_header_param(header_config),
                    )
                    validate_dataframe_size(df)

                    logger.warning(
                        "Used fallback encoding %s instead of %s", alt_encoding, encoding
                    )
                    await ctx.info(f"Used fallback encoding {alt_encoding} due to encoding error")
                    break
                except UnicodeDecodeError as fallback_error:
                    last_error = fallback_error
                    continue
                except Exception as other_error:
                    logger.debug("Failed with encoding %s: %s", alt_encoding, other_error)
                    continue
        else:
            msg = f"Encoding error with all attempted encodings: {last_error}. Try specifying a different encoding."
            raise ToolError(msg) from last_error

        if df is None:
            msg = f"Failed to download CSV with any encoding: {last_error}"

            raise ToolError(msg) from last_error

    await ctx.report_progress(0.8)

    # Get or create session
    session_manager = get_session_manager()
    session = session_manager.get_or_create_session(session_id)

    if df is None:
        msg = "Failed to load data from URL"
        raise ToolError(msg)

    session.load_data(df, url)

    await ctx.report_progress(1.0)
    await ctx.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from URL")

    # Create data preview with indices
    preview_data = create_data_preview_with_indices(df, 5)
    data_preview = DataPreview(
        rows=preview_data["records"],
        row_count=preview_data["total_rows"],
        column_count=preview_data["total_columns"],
        truncated=preview_data["preview_rows"] < preview_data["total_rows"],
    )

    return LoadResult(
        rows_affected=len(df),
        columns_affected=[str(col) for col in df.columns],
        data=data_preview,
        memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
    )


# Implementation: parses CSV from string using StringIO with pandas read_csv
# Validates content not empty, handles malformed CSV with specific error messages
# Supports header detection, quoted fields, automatic type inference
async def load_csv_from_content(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    content: Annotated[str, Field(description="CSV data as string content")],
    delimiter: Annotated[
        str, Field(description="Column delimiter character (comma, tab, semicolon, pipe)")
    ] = ",",
    *,
    header_config: Annotated[
        HeaderConfigUnion | None,
        Field(default=None, description="Header detection configuration"),
    ] = None,
) -> LoadResult:
    """Load CSV data from string content into DataBeak session.

    Parses CSV data directly from string with validation. Returns session ID and data preview for
    further operations.
    """
    # Get session_id from FastMCP context
    session_id = ctx.session_id

    await ctx.info("Loading CSV from content string")

    # Handle default header configuration
    if header_config is None:
        header_config = AutoDetectHeader()

    if not content or not content.strip():
        msg = "Content cannot be empty"
        raise ToolError(msg)

    # Parse CSV from string using StringIO
    try:
        df = pd.read_csv(
            StringIO(content),
            delimiter=delimiter,
            header=resolve_header_param(header_config),
        )
    except pd.errors.EmptyDataError as e:
        msg = "CSV content is empty or contains no data"
        raise ToolError(msg) from e
    except pd.errors.ParserError as e:
        msg = f"CSV parsing error: {e}"

        raise ToolError(msg) from e

    if df.empty:
        msg = "Parsed CSV contains no data rows"
        raise ToolError(msg)

    # Get or create session
    session_manager = get_session_manager()
    session = session_manager.get_or_create_session(session_id)
    session.load_data(df, None)

    await ctx.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from content")

    # Create data preview with indices
    preview_data = create_data_preview_with_indices(df, 5)
    data_preview = DataPreview(
        rows=preview_data["records"],
        row_count=preview_data["total_rows"],
        column_count=preview_data["total_columns"],
        truncated=preview_data["preview_rows"] < preview_data["total_rows"],
    )

    return LoadResult(
        rows_affected=len(df),
        columns_affected=[str(col) for col in df.columns],
        data=data_preview,
        memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
    )


# Implementation: retrieves session metadata from session manager
# Returns comprehensive info including timestamps, data status, auto-save config
# Essential for workflow coordination and session state verification
async def get_session_info(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
) -> SessionInfoResult:
    """Get comprehensive information about a specific session.

    Returns session metadata, data status, and configuration. Essential for session management and
    workflow coordination.
    """
    # Get session_id from FastMCP context
    session_id = ctx.session_id

    session = get_session_only(session_id)

    await ctx.info(f"Retrieved info for session {session_id}")

    # Get comprehensive session information
    info = session.get_info()

    return SessionInfoResult(
        created_at=info.created_at.isoformat(),
        last_modified=info.last_accessed.isoformat(),
        data_loaded=session.has_data(),
        row_count=info.row_count if session.has_data() else None,
        column_count=info.column_count if session.has_data() else None,
    )


# ============================================================================
# FASTMCP SERVER SETUP
# ============================================================================


# Create I/O server
io_server = FastMCP(
    "DataBeak-IO",
    instructions="I/O operations server for DataBeak with comprehensive CSV loading from web sources (URLs and string content)",
)


# Register the logic functions directly as MCP tools (no wrapper functions needed)
io_server.tool(name="load_csv_from_url")(load_csv_from_url)
io_server.tool(name="load_csv_from_content")(load_csv_from_content)
io_server.tool(name="get_session_info")(get_session_info)
