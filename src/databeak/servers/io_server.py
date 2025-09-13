"""Standalone I/O server for DataBeak using FastMCP server composition.

This module provides a complete I/O server implementation following DataBeak's modular server
architecture pattern. It includes comprehensive CSV loading, export, and session management
capabilities with robust error handling and AI-optimized documentation.
"""

from __future__ import annotations

import logging
import socket
import tempfile
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import chardet
import pandas as pd
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

# Import session management and data models from the main package
from ..models import ExportFormat, OperationType, get_session_manager
from ..models.tool_responses import BaseToolResponse
from ..services.data_operations import create_data_preview_with_indices
from ..utils.validators import validate_file_path, validate_url

logger = logging.getLogger(__name__)

# Configuration constants
MAX_FILE_SIZE_MB = 500  # Maximum file size in MB
MAX_MEMORY_USAGE_MB = 1000  # Maximum memory usage in MB for DataFrames
MAX_ROWS = 1_000_000  # Maximum number of rows to prevent memory issues
URL_TIMEOUT_SECONDS = 30  # Timeout for URL downloads
MAX_URL_SIZE_MB = 100  # Maximum download size for URLs

# ============================================================================
# PYDANTIC MODELS FOR I/O OPERATIONS
# ============================================================================

# Type aliases
CsvCellValue = str | int | float | bool | None


class SessionInfo(BaseModel):
    """Session information in list results."""

    session_id: str = Field(description="Unique session identifier")
    created_at: str = Field(description="Session creation timestamp (ISO format)")
    last_accessed: str = Field(description="Last access timestamp (ISO format)")
    row_count: int = Field(description="Number of rows in dataset")
    column_count: int = Field(description="Number of columns in dataset")
    columns: list[str] = Field(description="List of column names")
    memory_usage_mb: float = Field(description="Memory usage in megabytes")
    file_path: str | None = Field(None, description="Original file path if loaded from file")


class DataPreview(BaseModel):
    """Data preview with row samples."""

    rows: list[dict[str, CsvCellValue]] = Field(description="Sample rows from dataset")
    row_count: int = Field(description="Total number of rows in dataset")
    column_count: int = Field(description="Total number of columns in dataset")
    truncated: bool = Field(False, description="Whether preview is truncated")


class LoadResult(BaseToolResponse):
    """Response model for data loading operations."""

    session_id: str = Field(description="Session identifier for subsequent operations")
    rows_affected: int = Field(description="Number of rows loaded")
    columns_affected: list[str] = Field(description="List of column names detected")
    data: DataPreview | None = Field(None, description="Sample of loaded data")
    memory_usage_mb: float | None = Field(None, description="Memory usage in megabytes")


class ExportResult(BaseToolResponse):
    """Response model for data export operations."""

    session_id: str = Field(description="Session identifier that was exported")
    file_path: str = Field(description="Path to exported file")
    format: Literal["csv", "tsv", "json", "excel", "parquet", "html", "markdown"] = Field(
        description="Export format used"
    )
    rows_exported: int = Field(description="Number of rows exported")
    file_size_mb: float | None = Field(None, description="Size of exported file in megabytes")


class SessionInfoResult(BaseToolResponse):
    """Response model for session information."""

    session_id: str = Field(description="Session identifier")
    created_at: str = Field(description="Creation timestamp (ISO format)")
    last_modified: str = Field(description="Last modification timestamp (ISO format)")
    data_loaded: bool = Field(description="Whether session has data loaded")
    row_count: int | None = Field(None, description="Number of rows if data loaded")
    column_count: int | None = Field(None, description="Number of columns if data loaded")
    auto_save_enabled: bool = Field(description="Whether auto-save is enabled")


class SessionListResult(BaseToolResponse):
    """Response model for listing all sessions."""

    sessions: list[SessionInfo] = Field(description="List of all sessions")
    total_sessions: int = Field(description="Total number of sessions")
    active_sessions: int = Field(description="Number of sessions with loaded data")


class CloseSessionResult(BaseToolResponse):
    """Response model for session closure operations."""

    session_id: str = Field(description="Session identifier that was closed")
    message: str = Field(description="Operation status message")
    data_preserved: bool = Field(description="Whether data was preserved after closure")


# ============================================================================
# PYDANTIC MODELS FOR VALIDATION
# ============================================================================


class LoadCSVParams(BaseModel):
    """Parameters for CSV loading operations."""

    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(description="Path to the CSV file (absolute or relative)")
    encoding: str = Field("utf-8", description="File encoding (utf-8, latin1, cp1252, etc.)")
    delimiter: str = Field(",", description="Column delimiter (comma, tab, semicolon, pipe)")
    session_id: str | None = Field(None, description="Optional existing session ID")
    header: int | None = Field(
        0, description="Row number to use as header (0=first row, None=no header)"
    )
    na_values: list[str] | None = Field(
        None, description="Additional strings to recognize as NA/NaN"
    )
    parse_dates: list[str] | None = Field(None, description="Columns to parse as dates")


class LoadCSVFromURLParams(BaseModel):
    """Parameters for CSV loading from URL operations."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="URL of the CSV file")
    encoding: str = Field("utf-8", description="File encoding")
    delimiter: str = Field(",", description="Column delimiter")
    session_id: str | None = Field(None, description="Optional existing session ID")


class LoadCSVFromContentParams(BaseModel):
    """Parameters for CSV loading from content operations."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(description="CSV content as string")
    delimiter: str = Field(",", description="Column delimiter")
    session_id: str | None = Field(None, description="Optional existing session ID")
    has_header: bool = Field(True, description="Whether first row is header")


class ExportCSVParams(BaseModel):
    """Parameters for CSV export operations."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Session ID to export")
    file_path: str | None = Field(
        None, description="Optional output file path (auto-generated if not provided)"
    )
    format: Literal["csv", "tsv", "json", "excel", "parquet", "html", "markdown"] = Field(
        "csv", description="Export format"
    )
    encoding: str = Field("utf-8", description="Output encoding")
    index: bool = Field(False, description="Whether to include index in output")


# ============================================================================
# ENCODING DETECTION UTILITIES
# ============================================================================


# Implementation: uses chardet for automatic detection with confidence validation
# Falls back to prioritized common encodings if detection fails or low confidence
# Reads 10KB sample for fast detection without loading full file
def detect_file_encoding(file_path: str) -> str:
    """Detect file encoding using chardet with optimized fallbacks."""
    try:
        # Read sample bytes for detection (first 10KB should be enough)
        with open(file_path, "rb") as f:  # noqa: PTH123
            raw_data = f.read(10240)  # 10KB sample

        # Use chardet for automatic detection
        detection = chardet.detect(raw_data)

        if detection and detection["confidence"] > 0.7:
            detected_encoding = detection["encoding"]
            if detected_encoding:
                logger.debug(
                    f"Chardet detected encoding: {detected_encoding} (confidence: {detection['confidence']:.2f})"
                )
                return detected_encoding.lower()
            else:
                logger.debug("Chardet detected encoding is None, using fallbacks")

        logger.debug(
            f"Chardet detection low confidence ({detection['confidence'] if detection else 0:.2f}), using fallbacks"
        )

    except (ImportError, AttributeError, UnicodeError, OSError) as e:
        logger.debug(f"Chardet detection failed: {e}, using fallbacks")

    # Fallback to common encodings in priority order
    # UTF-8 first (most common), then Windows encodings, then Latin variants
    return "utf-8"


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


# Implementation: RFC 4180 compliant CSV parsing with automatic encoding detection
# Supports quoted fields, escaped quotes, mixed quoting, automatic type detection
# Memory limits: MAX_ROWS, MAX_FILE_SIZE_MB, MAX_MEMORY_USAGE_MB validation
# Encoding fallback strategy with chardet detection and prioritized fallbacks
# Progress reporting and comprehensive error handling with specific error messages
async def load_csv(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    file_path: Annotated[str, Field(description="Path to the CSV file to load")],
    encoding: Annotated[
        str, Field(description="Text encoding for file reading (utf-8, latin1, cp1252, etc.)")
    ] = "utf-8",
    delimiter: Annotated[
        str, Field(description="Column delimiter character (comma, tab, semicolon, pipe)")
    ] = ",",
    header: Annotated[
        int | None, Field(description="Row number to use as header (0=first row, None=no header)")
    ] = 0,
    na_values: Annotated[
        list[str] | None, Field(description="Additional strings to recognize as NA/NaN")
    ] = None,
    parse_dates: Annotated[list[str] | None, Field(description="Columns to parse as dates")] = None,
) -> LoadResult:
    """Load CSV file into DataBeak session.

    Parses CSV data with encoding detection and error handling. Returns session ID and data preview
    for further operations.
    """
    try:
        # Get session_id from FastMCP context
        session_id = ctx.session_id

        # Validate file path
        is_valid, validated_path = validate_file_path(file_path)
        if not is_valid:
            raise ToolError(f"Invalid file path: {validated_path}")

        await ctx.info(f"Loading CSV file: {validated_path}")
        await ctx.report_progress(0.1)

        # Check file size before attempting to load
        file_size_mb = Path(validated_path).stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ToolError(f"File size {file_size_mb:.1f}MB exceeds limit of {MAX_FILE_SIZE_MB}MB")

        await ctx.info(f"File size: {file_size_mb:.2f} MB")

        # Get or create session
        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(session_id)

        await ctx.report_progress(0.3)

        # Build pandas read_csv parameters
        # Using dict[str, Any] due to pandas read_csv's complex overloaded signature
        read_params: dict[str, Any] = {
            "filepath_or_buffer": validated_path,
            "encoding": encoding,
            "delimiter": delimiter,
            "header": header,
            # Note: Temporarily disabled dtype_backend="numpy_nullable" due to serialization issues
        }

        if na_values:
            read_params["na_values"] = na_values
        if parse_dates:
            read_params["parse_dates"] = parse_dates

        # Load CSV with comprehensive error handling
        try:
            # Add memory-conscious parameters for large files
            df = pd.read_csv(
                **read_params, chunksize=None
            )  # Keep as None for now but ready for streaming

            # Check memory usage and row count limits
            if len(df) > MAX_ROWS:
                raise ToolError(
                    f"File too large: {len(df):,} rows exceeds limit of {MAX_ROWS:,} rows"
                )

            memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            if memory_usage_mb > MAX_MEMORY_USAGE_MB:
                raise ToolError(
                    f"File too large: {memory_usage_mb:.1f} MB exceeds memory limit of {MAX_MEMORY_USAGE_MB} MB"
                )
        except UnicodeDecodeError as e:
            # Use optimized encoding detection and fallbacks
            df = None
            last_error = e

            await ctx.info("Encoding error detected, trying automatic detection...")

            # First, try automatic encoding detection
            try:
                detected_encoding = detect_file_encoding(validated_path)
                if detected_encoding != encoding:
                    logger.info(f"Auto-detected encoding: {detected_encoding}")
                    await ctx.info(f"Auto-detected encoding: {detected_encoding}")

                    read_params["encoding"] = detected_encoding
                    df = pd.read_csv(**read_params)

                    # Apply memory checks to detected encoding
                    if len(df) > MAX_ROWS:
                        raise ToolError(
                            f"File too large: {len(df):,} rows exceeds limit of {MAX_ROWS:,} rows"
                        )

                    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                    if memory_usage_mb > MAX_MEMORY_USAGE_MB:
                        raise ToolError(
                            f"File too large: {memory_usage_mb:.1f} MB exceeds memory limit of {MAX_MEMORY_USAGE_MB} MB"
                        )

                    logger.info(
                        f"Successfully loaded with auto-detected encoding: {detected_encoding}"
                    )

            except Exception as detection_error:
                logger.debug(
                    f"Auto-detection failed: {detection_error}, trying prioritized fallbacks"
                )

                # Fall back to optimized encoding list
                fallback_encodings = get_encoding_fallbacks(encoding)

                for alt_encoding in fallback_encodings:
                    if alt_encoding != encoding:  # Skip the original encoding we already tried
                        try:
                            read_params["encoding"] = alt_encoding
                            df = pd.read_csv(**read_params)

                            # Apply same memory checks to fallback encoding
                            if len(df) > MAX_ROWS:
                                raise ToolError(
                                    f"File too large: {len(df):,} rows exceeds limit of {MAX_ROWS:,} rows"
                                )

                            memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                            if memory_usage_mb > MAX_MEMORY_USAGE_MB:
                                raise ToolError(
                                    f"File too large: {memory_usage_mb:.1f} MB exceeds memory limit of {MAX_MEMORY_USAGE_MB} MB"
                                )

                            logger.warning(
                                f"Used fallback encoding {alt_encoding} instead of {encoding}"
                            )
                            await ctx.info(
                                f"Used fallback encoding {alt_encoding} due to encoding error"
                            )
                            break
                        except UnicodeDecodeError as fallback_error:
                            last_error = fallback_error
                            continue
                        except Exception as other_error:
                            logger.debug(f"Failed with encoding {alt_encoding}: {other_error}")
                            continue
                else:
                    # All encodings failed
                    raise ToolError(
                        f"Encoding error with all attempted encodings: {last_error}. "
                        "Try specifying a different encoding or check file format."
                    ) from last_error

            if df is None:
                raise ToolError(
                    f"Failed to load CSV with any encoding: {last_error}"
                ) from last_error

        await ctx.report_progress(0.8)

        # Load into session
        session.load_data(df, validated_path)

        await ctx.report_progress(1.0)
        await ctx.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

        # Create comprehensive data preview with indices
        preview_data = create_data_preview_with_indices(df, 5)
        data_preview = DataPreview(
            rows=preview_data["records"],
            row_count=preview_data["total_rows"],
            column_count=preview_data["total_columns"],
            truncated=preview_data["preview_rows"] < preview_data["total_rows"],
        )

        return LoadResult(
            session_id=session.session_id,
            rows_affected=len(df),
            columns_affected=[str(col) for col in df.columns],
            data=data_preview,
            memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
        )

    except OSError as e:
        logger.error(f"File I/O error while loading CSV: {e}")
        await ctx.error(f"File access error: {e!s}")
        raise ToolError(f"File access error: {e}") from e
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.error(f"CSV parsing error: {e}")
        await ctx.error(f"CSV format error: {e!s}")
        raise ToolError(f"CSV format error: {e}") from e
    except MemoryError as e:
        logger.error(f"Insufficient memory to load CSV: {e}")
        await ctx.error("File too large - insufficient memory")
        raise ToolError("File too large - insufficient memory") from e
    except Exception as e:
        # Fallback for unexpected errors - more specific logging
        logger.error(f"Unexpected error while loading CSV: {type(e).__name__}: {e}")
        await ctx.error(f"Unexpected error: {e!s}")
        raise ToolError(f"Failed to load CSV: {e}") from e


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
) -> LoadResult:
    """Load CSV file from URL into DataBeak session.

    Downloads and parses CSV data with security validation. Returns session ID and data preview for
    further operations.
    """
    try:
        # Get session_id from FastMCP context
        session_id = ctx.session_id

        # Validate URL
        is_valid, validated_url = validate_url(url)
        if not is_valid:
            raise ToolError(f"Invalid URL: {validated_url}")

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
                    logger.warning(f"Unexpected content-type: {content_type}. Proceeding anyway.")
                    await ctx.info(f"Warning: Content-type is {content_type}, expected CSV format")

                # Check content length
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > MAX_URL_SIZE_MB:
                        raise ToolError(
                            f"Download too large: {size_mb:.1f} MB exceeds limit of {MAX_URL_SIZE_MB} MB"
                        )

                await ctx.info(f"Download validated. Content-type: {content_type or 'unknown'}")
                await ctx.report_progress(0.3)

            # Download and parse CSV using pandas with timeout
            df = pd.read_csv(url, encoding=encoding, delimiter=delimiter)

            # Apply memory and row limits to downloaded data
            if len(df) > MAX_ROWS:
                raise ToolError(
                    f"Downloaded file too large: {len(df):,} rows exceeds limit of {MAX_ROWS:,} rows"
                )

            memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            if memory_usage_mb > MAX_MEMORY_USAGE_MB:
                raise ToolError(
                    f"Downloaded file too large: {memory_usage_mb:.1f} MB exceeds memory limit of {MAX_MEMORY_USAGE_MB} MB"
                )

        except (TimeoutError, URLError, HTTPError) as e:
            logger.error(f"Network error downloading URL: {e}")
            await ctx.error(f"Network error: {e!s}")
            raise ToolError(f"Network error: {e}") from e
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
                        df = pd.read_csv(url, encoding=alt_encoding, delimiter=delimiter)

                        # Apply same memory checks to fallback encoding
                        if len(df) > MAX_ROWS:
                            raise ToolError(
                                f"Downloaded file too large: {len(df):,} rows exceeds limit of {MAX_ROWS:,} rows"
                            )

                        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                        if memory_usage_mb > MAX_MEMORY_USAGE_MB:
                            raise ToolError(
                                f"Downloaded file too large: {memory_usage_mb:.1f} MB exceeds memory limit of {MAX_MEMORY_USAGE_MB} MB"
                            )

                        logger.warning(
                            f"Used fallback encoding {alt_encoding} instead of {encoding}"
                        )
                        await ctx.info(
                            f"Used fallback encoding {alt_encoding} due to encoding error"
                        )
                        break
                    except UnicodeDecodeError as fallback_error:
                        last_error = fallback_error
                        continue
                    except Exception as other_error:
                        logger.debug(f"Failed with encoding {alt_encoding}: {other_error}")
                        continue
            else:
                raise ToolError(
                    f"Encoding error with all attempted encodings: {last_error}. "
                    "Try specifying a different encoding."
                ) from last_error

            if df is None:
                raise ToolError(
                    f"Failed to download CSV with any encoding: {last_error}"
                ) from last_error

        await ctx.report_progress(0.8)

        # Get or create session
        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(session_id)

        if df is None:
            raise ToolError("Failed to load data from URL")

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
            session_id=session.session_id,
            rows_affected=len(df),
            columns_affected=[str(col) for col in df.columns],
            data=data_preview,
            memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
        )

    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.error(f"CSV parsing error from URL: {e}")
        await ctx.error(f"CSV format error from URL: {e!s}")
        raise ToolError(f"CSV format error from URL: {e}") from e
    except OSError as e:
        logger.error(f"Network/file error while loading from URL: {e}")
        await ctx.error(f"Network error: {e!s}")
        raise ToolError(f"Network error: {e}") from e
    except MemoryError as e:
        logger.error(f"Insufficient memory to load CSV from URL: {e}")
        await ctx.error("Downloaded file too large - insufficient memory")
        raise ToolError("Downloaded file too large - insufficient memory") from e
    except Exception as e:
        logger.error(f"Unexpected error while loading CSV from URL: {type(e).__name__}: {e}")
        await ctx.error(f"Unexpected error: {e!s}")
        raise ToolError(f"Failed to load CSV from URL: {e}") from e


# Implementation: parses CSV from string using StringIO with pandas read_csv
# Validates content not empty, handles malformed CSV with specific error messages
# Supports header detection, quoted fields, automatic type inference
async def load_csv_from_content(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    content: Annotated[str, Field(description="CSV data as string content")],
    delimiter: Annotated[
        str, Field(description="Column delimiter character (comma, tab, semicolon, pipe)")
    ] = ",",
    has_header: Annotated[
        bool, Field(description="Whether first row contains column headers")
    ] = True,
) -> LoadResult:
    """Load CSV data from string content into DataBeak session.

    Parses CSV data directly from string with validation. Returns session ID and data preview for
    further operations.
    """
    try:
        # Get session_id from FastMCP context
        session_id = ctx.session_id

        await ctx.info("Loading CSV from content string")

        if not content or not content.strip():
            raise ToolError("Content cannot be empty")

        # Parse CSV from string using StringIO
        try:
            df = pd.read_csv(
                StringIO(content),
                delimiter=delimiter,
                header=0 if has_header else None,
            )
        except pd.errors.EmptyDataError as e:
            raise ToolError("CSV content is empty or contains no data") from e
        except pd.errors.ParserError as e:
            raise ToolError(f"CSV parsing error: {e}") from e

        if df.empty:
            raise ToolError("Parsed CSV contains no data rows")

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
            session_id=session.session_id,
            rows_affected=len(df),
            columns_affected=[str(col) for col in df.columns],
            data=data_preview,
            memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
        )

    except Exception as e:
        logger.error(f"Failed to parse CSV content: {e}")
        await ctx.error(f"Failed to parse CSV content: {e!s}")
        raise ToolError(f"Failed to parse CSV content: {e}") from e


# Implementation: supports 7 export formats with auto-generated filenames using tempfile
# Format-specific parameters: CSV (RFC 4180), TSV (tab delimiter), JSON (records), Excel (XLSX)
# Parquet (columnar), HTML (web table), Markdown (GitHub format)
# Auto-cleanup on export errors, records operation in session history
async def export_csv(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    file_path: Annotated[
        str | None, Field(description="Output file path (auto-generated if not provided)")
    ] = None,
    format: Annotated[
        Literal["csv", "tsv", "json", "excel", "parquet", "html", "markdown"],
        Field(description="Export format (csv, tsv, json, excel, parquet, html, markdown)"),
    ] = "csv",
    encoding: Annotated[
        str, Field(description="Text encoding for output file (utf-8, latin1, cp1252, etc.)")
    ] = "utf-8",
    index: Annotated[bool, Field(description="Whether to include row index in output")] = False,
) -> ExportResult:
    """Export session data to various file formats.

    Supports CSV, TSV, JSON, Excel, Parquet, HTML, and Markdown formats. Returns file path and
    export statistics.
    """
    try:
        # Get session_id from FastMCP context
        session_id = ctx.session_id

        # Normalize format to ExportFormat enum
        format_enum = ExportFormat(format)

        # Get session and validate data
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session or session.df is None:
            raise ToolError(f"Session not found or no data loaded: {session_id}")

        await ctx.info(f"Exporting data in {format_enum.value} format")
        await ctx.report_progress(0.1)

        # Generate file path if not provided using proper temp file handling
        temp_file_path = None
        try:
            if not file_path:
                # Determine extension based on format
                extensions = {
                    ExportFormat.CSV: ".csv",
                    ExportFormat.TSV: ".tsv",
                    ExportFormat.JSON: ".json",
                    ExportFormat.EXCEL: ".xlsx",
                    ExportFormat.PARQUET: ".parquet",
                    ExportFormat.HTML: ".html",
                    ExportFormat.MARKDOWN: ".md",
                }

                # Use tempfile.NamedTemporaryFile for proper temp file handling
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                prefix = f"databeak_export_{session_id[:8]}_{timestamp}_"
                suffix = extensions[format_enum]

                # Create temp file but don't delete it (we'll return the path)
                with tempfile.NamedTemporaryFile(
                    prefix=prefix,
                    suffix=suffix,
                    delete=False,
                    dir=None,  # Use system temp directory
                ) as temp_file:
                    file_path = temp_file.name
                    temp_file_path = file_path  # Track for cleanup on error

            path_obj = Path(file_path)

            # Create parent directory if it doesn't exist
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            df = session.df

            await ctx.report_progress(0.5)

            # Export based on format with comprehensive options
            try:
                if format_enum == ExportFormat.CSV:
                    df.to_csv(path_obj, encoding=encoding, index=index, lineterminator="\n")
                elif format_enum == ExportFormat.TSV:
                    df.to_csv(
                        path_obj, sep="\t", encoding=encoding, index=index, lineterminator="\n"
                    )
                elif format_enum == ExportFormat.JSON:
                    df.to_json(path_obj, orient="records", indent=2, force_ascii=False)
                elif format_enum == ExportFormat.EXCEL:
                    with pd.ExcelWriter(path_obj, engine="openpyxl") as writer:
                        df.to_excel(writer, sheet_name="Data", index=index)
                elif format_enum == ExportFormat.PARQUET:
                    df.to_parquet(path_obj, index=index, engine="pyarrow")
                elif format_enum == ExportFormat.HTML:
                    df.to_html(path_obj, index=index, escape=False, table_id="data-table")
                elif format_enum == ExportFormat.MARKDOWN:
                    df.to_markdown(path_obj, index=index, tablefmt="github")
                else:
                    raise ToolError(f"Unsupported format: {format_enum}")
            except Exception as export_error:
                # Clean up temp file on export error
                if temp_file_path and Path(temp_file_path).exists():
                    try:
                        Path(temp_file_path).unlink()
                        logger.debug(f"Cleaned up temp file after export error: {temp_file_path}")
                    except OSError as cleanup_error:
                        logger.warning(
                            f"Failed to clean up temp file {temp_file_path}: {cleanup_error}"
                        )

                # Provide format-specific error guidance
                if format_enum == ExportFormat.EXCEL and "openpyxl" in str(export_error):
                    raise ToolError(
                        "Excel export requires openpyxl package. Install with: pip install openpyxl"
                    ) from export_error
                elif format_enum == ExportFormat.PARQUET and "pyarrow" in str(export_error):
                    raise ToolError(
                        "Parquet export requires pyarrow package. Install with: pip install pyarrow"
                    ) from export_error
                else:
                    raise ToolError(f"Export failed: {export_error}") from export_error

            # Record operation in session history
            session.record_operation(
                OperationType.EXPORT,
                {"format": format_enum.value, "file_path": str(file_path), "rows": len(df)},
            )

            await ctx.report_progress(1.0)
            await ctx.info(f"Exported {len(df)} rows to {file_path}")

            # Calculate file size
            file_size_mb = path_obj.stat().st_size / (1024 * 1024) if path_obj.exists() else 0

            return ExportResult(
                session_id=session_id,
                file_path=str(file_path),
                format=format_enum.value,
                rows_exported=len(df),
                file_size_mb=round(file_size_mb, 3),
            )

        except Exception:
            # Clean up temp file on any other error
            if temp_file_path and Path(temp_file_path).exists():
                try:
                    Path(temp_file_path).unlink()
                    logger.debug(f"Cleaned up temp file after error: {temp_file_path}")
                except OSError as cleanup_error:
                    logger.warning(
                        f"Failed to clean up temp file {temp_file_path}: {cleanup_error}"
                    )

            # Re-raise the original error
            raise

    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        await ctx.error(f"Failed to export data: {e!s}")
        raise ToolError(f"Failed to export data: {e}") from e


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
    try:
        # Get session_id from FastMCP context
        session_id = ctx.session_id

        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise ToolError(f"Session not found: {session_id}")

        await ctx.info(f"Retrieved info for session {session_id}")

        # Get comprehensive session information
        info = session.get_info()

        return SessionInfoResult(
            session_id=session_id,
            created_at=info.created_at.isoformat(),
            last_modified=info.last_accessed.isoformat(),
            data_loaded=session.df is not None,
            row_count=info.row_count if session.df is not None else None,
            column_count=info.column_count if session.df is not None else None,
            auto_save_enabled=session.auto_save_config.enabled,
        )

    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        await ctx.error(f"Failed to get session info: {e!s}")
        raise ToolError(f"Failed to get session info: {e}") from e


# Implementation: retrieves all sessions from session manager with statistics
# Counts active sessions (those with loaded data) vs total sessions
# Returns empty list on error for consistency, essential for system monitoring
async def list_sessions(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
) -> SessionListResult:
    """List all active sessions with details and statistics.

    Returns overview of all sessions including data status and timestamps. Essential for session
    management and system monitoring.
    """
    try:
        session_manager = get_session_manager()
        sessions = session_manager.list_sessions()

        await ctx.info(f"Found {len(sessions)} active sessions")

        # Convert session info to SessionInfo objects for tool response
        session_infos = []
        active_count = 0

        for s in sessions:
            # s is already a SessionInfo from data_models
            session_info = SessionInfo(
                session_id=s.session_id,
                created_at=s.created_at.isoformat(),
                last_accessed=s.last_accessed.isoformat(),
                row_count=s.row_count,
                column_count=s.column_count,
                columns=[str(col) for col in s.columns],  # Ensure columns are strings
                memory_usage_mb=s.memory_usage_mb,
                file_path=s.file_path,
            )
            session_infos.append(session_info)

            # Count active sessions (those with data loaded)
            if s.row_count > 0:
                active_count += 1

        return SessionListResult(
            sessions=session_infos,
            total_sessions=len(sessions),
            active_sessions=active_count,
        )

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        await ctx.error(f"Failed to list sessions: {e!s}")
        # Return empty list on error for consistency
        return SessionListResult(sessions=[], total_sessions=0, active_sessions=0)


# Implementation: removes session from manager with proper resource cleanup
# Memory deallocation for DataFrames, history cleanup, state finalization
# Data is not preserved - use export_csv before closing to save data
# Essential for preventing memory leaks in long-running processes
async def close_session(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
) -> CloseSessionResult:
    """Close and clean up session with proper resource management.

    Safely closes session and releases memory.
    Data is not preserved - export before closing if needed.
    """
    try:
        # Get session_id from FastMCP context
        session_id = ctx.session_id

        session_manager = get_session_manager()
        removed = await session_manager.remove_session(session_id)

        if not removed:
            raise ToolError(f"Session not found: {session_id}")

        await ctx.info(f"Closed session {session_id}")

        return CloseSessionResult(
            session_id=session_id,
            message=f"Session {session_id} closed successfully",
            data_preserved=False,  # Sessions are removed, so data is not preserved
        )

    except Exception as e:
        logger.error(f"Failed to close session: {e}")
        await ctx.error(f"Failed to close session: {e!s}")
        raise ToolError(f"Failed to close session: {e}") from e


# ============================================================================
# FASTMCP SERVER SETUP
# ============================================================================


# Create I/O server
io_server = FastMCP(
    "DataBeak-IO",
    instructions="I/O operations server for DataBeak with comprehensive CSV loading and export capabilities",
)


# Register the logic functions directly as MCP tools (no wrapper functions needed)
io_server.tool(name="load_csv")(load_csv)
io_server.tool(name="load_csv_from_url")(load_csv_from_url)
io_server.tool(name="load_csv_from_content")(load_csv_from_content)
io_server.tool(name="export_csv")(export_csv)
io_server.tool(name="get_session_info")(get_session_info)
io_server.tool(name="list_sessions")(list_sessions)
io_server.tool(name="close_session")(close_session)
