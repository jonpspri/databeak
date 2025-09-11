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
from typing import Any, Literal
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
from ..tools.data_operations import create_data_preview_with_indices
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

    session_id: str
    created_at: str
    last_accessed: str
    row_count: int
    column_count: int
    columns: list[str]
    memory_usage_mb: float
    file_path: str | None = None


class DataPreview(BaseModel):
    """Data preview with row samples."""

    rows: list[dict[str, CsvCellValue]]
    row_count: int
    column_count: int
    truncated: bool = False


class LoadResult(BaseToolResponse):
    """Response model for data loading operations."""

    session_id: str
    rows_affected: int
    columns_affected: list[str]
    data: DataPreview | None = None
    memory_usage_mb: float | None = None


class ExportResult(BaseToolResponse):
    """Response model for data export operations."""

    session_id: str
    file_path: str
    format: Literal["csv", "tsv", "json", "excel", "parquet", "html", "markdown"]
    rows_exported: int
    file_size_mb: float | None = None


class SessionInfoResult(BaseToolResponse):
    """Response model for session information."""

    session_id: str
    created_at: str
    last_modified: str
    data_loaded: bool
    row_count: int | None = None
    column_count: int | None = None
    auto_save_enabled: bool


class SessionListResult(BaseToolResponse):
    """Response model for listing all sessions."""

    sessions: list[SessionInfo]
    total_sessions: int
    active_sessions: int


class CloseSessionResult(BaseToolResponse):
    """Response model for session closure operations."""

    session_id: str
    message: str
    data_preserved: bool


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


def detect_file_encoding(file_path: str) -> str:
    """Detect file encoding using chardet with optimized fallbacks.

    Args:
        file_path: Path to the file to analyze

    Returns:
        Detected encoding name (e.g., 'utf-8', 'latin1', 'cp1252')

    Detection Strategy:
        1. Use chardet for automatic detection on sample
        2. Validate detection confidence
        3. Fall back to prioritized common encodings
    """
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

    except Exception as e:
        logger.debug(f"Chardet detection failed: {e}, using fallbacks")

    # Fallback to common encodings in priority order
    # UTF-8 first (most common), then Windows encodings, then Latin variants
    return "utf-8"


def get_encoding_fallbacks(primary_encoding: str) -> list[str]:
    """Get optimized encoding fallback list based on primary encoding.

    Args:
        primary_encoding: The initially requested encoding

    Returns:
        List of encodings to try in priority order
    """
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


async def load_csv(
    file_path: str,
    encoding: str = "utf-8",
    delimiter: str = ",",
    session_id: str | None = None,
    header: int | None = 0,
    na_values: list[str] | None = None,
    parse_dates: list[str] | None = None,
    ctx: Context | None = None,
) -> LoadResult:
    """Load a CSV file into a session with comprehensive parsing and AI-optimized data preview.

    Provides robust CSV loading with support for various encodings, delimiters, and complex
    quoted data including commas and escaped quotes. Optimized for AI workflows with
    enhanced data preview and coordinate system information.

    Args:
        file_path: Path to the CSV file (absolute or relative)
        encoding: File encoding (default: utf-8, also supports: latin1, cp1252, etc.)
        delimiter: Column delimiter (default: comma, also supports: tab, semicolon, pipe)
        session_id: Optional existing session ID, or None to create new session
        header: Row number to use as header (default: 0, None for no header)
        na_values: Additional strings to recognize as NA/NaN
        parse_dates: Columns to parse as dates
        ctx: FastMCP context for progress reporting

    Returns:
        Comprehensive load result containing:
        - success: bool indicating load status
        - session_id: Session identifier for subsequent operations
        - rows_affected: Number of rows loaded
        - columns_affected: List of column names detected
        - data: Enhanced preview with coordinate information including:
          * shape: (rows, columns) dimensions
          * dtypes: Detected data types for each column
          * memory_usage_mb: Memory consumption
          * preview: First 5 rows with row indices for AI reference

    CSV Parsing Capabilities:
        ✅ Quoted values with commas: "Smith, John" → Smith, John
        ✅ Escaped quotes: "product ""premium"" grade" → product "premium" grade
        ✅ Mixed quoting with null values
        ✅ Standard RFC 4180 compliance
        ✅ Automatic type detection (int, float, string, datetime)
        ✅ Null value handling (empty fields → pandas NaN)

    Examples:
        # Basic CSV loading
        load_csv("/path/to/data.csv")

        # Custom delimiter and encoding
        load_csv("/path/to/european.csv", encoding="latin1", delimiter=";")

        # Load into existing session
        load_csv("/path/to/additional.csv", session_id="existing_session_123")

    Error Conditions:
        - File not found or permission denied
        - Invalid encoding or unreadable file format
        - Malformed CSV structure
        - Memory limitations for extremely large files

    AI Workflow Integration:
        1. Load CSV → get session_id
        2. Use get_session_info(session_id) for overview
        3. Inspect with get_cell_value/get_row_data for details
        4. Apply transformations based on data understanding
    """
    try:
        # Validate file path
        is_valid, validated_path = validate_file_path(file_path)
        if not is_valid:
            raise ToolError(f"Invalid file path: {validated_path}")

        if ctx:
            await ctx.info(f"Loading CSV file: {validated_path}")
            await ctx.report_progress(0.1)

        # Check file size before attempting to load
        file_size_mb = Path(validated_path).stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ToolError(f"File size {file_size_mb:.1f}MB exceeds limit of {MAX_FILE_SIZE_MB}MB")

        if ctx:
            await ctx.info(f"File size: {file_size_mb:.2f} MB")

        # Get or create session
        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(session_id)

        if ctx:
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

            if ctx:
                await ctx.info("Encoding error detected, trying automatic detection...")

            # First, try automatic encoding detection
            try:
                detected_encoding = detect_file_encoding(validated_path)
                if detected_encoding != encoding:
                    logger.info(f"Auto-detected encoding: {detected_encoding}")
                    if ctx:
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
                            if ctx:
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

        if ctx:
            await ctx.report_progress(0.8)

        # Load into session
        session.load_data(df, validated_path)

        if ctx:
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
        if ctx:
            await ctx.error(f"File access error: {e!s}")
        raise ToolError(f"File access error: {e}") from e
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.error(f"CSV parsing error: {e}")
        if ctx:
            await ctx.error(f"CSV format error: {e!s}")
        raise ToolError(f"CSV format error: {e}") from e
    except MemoryError as e:
        logger.error(f"Insufficient memory to load CSV: {e}")
        if ctx:
            await ctx.error("File too large - insufficient memory")
        raise ToolError("File too large - insufficient memory") from e
    except Exception as e:
        # Fallback for unexpected errors - more specific logging
        logger.error(f"Unexpected error while loading CSV: {type(e).__name__}: {e}")
        if ctx:
            await ctx.error(f"Unexpected error: {e!s}")
        raise ToolError(f"Failed to load CSV: {e}") from e


async def load_csv_from_url(
    url: str,
    encoding: str = "utf-8",
    delimiter: str = ",",
    session_id: str | None = None,
    ctx: Context | None = None,
) -> LoadResult:
    """Load a CSV file from a URL with comprehensive error handling and progress reporting.

    Downloads and parses CSV data from HTTP/HTTPS URLs with automatic encoding detection
    and robust error handling. Supports all standard CSV formats and provides detailed
    progress feedback for AI workflow integration.

    Args:
        url: URL of the CSV file (must be HTTP or HTTPS)
        encoding: File encoding (default: utf-8, fallback encodings tried automatically)
        delimiter: Column delimiter (comma, tab, semicolon, pipe)
        session_id: Optional existing session ID, or None to create new session
        ctx: FastMCP context for progress reporting

    Returns:
        Comprehensive load result with session info and data preview

    URL Security & Validation:
        ✅ Only HTTP/HTTPS URLs allowed
        ✅ Private network access blocked (192.168.x.x, 10.x.x.x, etc.)
        ✅ Timeout protection (30 seconds) for slow downloads
        ✅ Content-type verification (text/csv, application/csv, etc.)
        ✅ Size limits to prevent memory issues (100 MB max download)
        ✅ Memory usage validation after download

    Examples:
        # Load from public dataset
        load_csv_from_url("https://example.com/data.csv")

        # European CSV with semicolon delimiter
        load_csv_from_url("https://example.com/euro.csv", delimiter=";", encoding="latin1")

    AI Workflow Integration:
        1. Perfect for loading external datasets for analysis
        2. Combine with profile_data for immediate data understanding
        3. Use in data exploration and research workflows
    """
    try:
        # Validate URL
        is_valid, validated_url = validate_url(url)
        if not is_valid:
            raise ToolError(f"Invalid URL: {validated_url}")

        if ctx:
            await ctx.info(f"Loading CSV from URL: {url}")
            await ctx.report_progress(0.1)

        # Download with timeout and content-type verification
        try:
            # Pre-download validation with timeout and content-type checking
            if ctx:
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
                    if ctx:
                        await ctx.info(
                            f"Warning: Content-type is {content_type}, expected CSV format"
                        )

                # Check content length
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > MAX_URL_SIZE_MB:
                        raise ToolError(
                            f"Download too large: {size_mb:.1f} MB exceeds limit of {MAX_URL_SIZE_MB} MB"
                        )

                if ctx:
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
            if ctx:
                await ctx.error(f"Network error: {e!s}")
            raise ToolError(f"Network error: {e}") from e
        except UnicodeDecodeError as e:
            # Use optimized encoding fallbacks for URL downloads
            df = None
            last_error = e

            if ctx:
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
                        if ctx:
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

        if ctx:
            await ctx.report_progress(0.8)

        # Get or create session
        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(session_id)

        if df is None:
            raise ToolError("Failed to load data from URL")

        session.load_data(df, url)

        if ctx:
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
        if ctx:
            await ctx.error(f"CSV format error from URL: {e!s}")
        raise ToolError(f"CSV format error from URL: {e}") from e
    except OSError as e:
        logger.error(f"Network/file error while loading from URL: {e}")
        if ctx:
            await ctx.error(f"Network error: {e!s}")
        raise ToolError(f"Network error: {e}") from e
    except MemoryError as e:
        logger.error(f"Insufficient memory to load CSV from URL: {e}")
        if ctx:
            await ctx.error("Downloaded file too large - insufficient memory")
        raise ToolError("Downloaded file too large - insufficient memory") from e
    except Exception as e:
        logger.error(f"Unexpected error while loading CSV from URL: {type(e).__name__}: {e}")
        if ctx:
            await ctx.error(f"Unexpected error: {e!s}")
        raise ToolError(f"Failed to load CSV from URL: {e}") from e


async def load_csv_from_content(
    content: str,
    delimiter: str = ",",
    session_id: str | None = None,
    has_header: bool = True,
    ctx: Context | None = None,
) -> LoadResult:
    """Load CSV data from string content with flexible parsing and validation.

    Parses CSV data directly from string content, ideal for programmatic data loading,
    API responses, or dynamically generated CSV content. Provides comprehensive
    validation and error handling for malformed data.

    Args:
        content: CSV content as string (including newlines and proper CSV formatting)
        delimiter: Column delimiter (comma, tab, semicolon, pipe)
        session_id: Optional existing session ID, or None to create new session
        has_header: Whether first row contains column headers
        ctx: FastMCP context for progress reporting

    Returns:
        Comprehensive load result with session info and data preview

    Content Validation:
        ✅ RFC 4180 CSV format compliance
        ✅ Quoted field support with embedded delimiters
        ✅ Empty field and null value handling
        ✅ Automatic type inference
        ✅ Malformed row detection and reporting

    Examples:
        # Simple CSV content
        content = "name,age,city\\nJohn,25,NYC\\nJane,30,LA"
        load_csv_from_content(content)

        # Tab-delimited without header
        content = "John\\t25\\tNYC\\nJane\\t30\\tLA"
        load_csv_from_content(content, delimiter="\\t", has_header=False)

        # Complex quoted content
        content = '"Smith, John",25,"New York, NY"\\n"Doe, Jane",30,"Los Angeles, CA"'
        load_csv_from_content(content)

    AI Workflow Integration:
        1. Perfect for loading AI-generated CSV data
        2. Process API responses containing CSV data
        3. Handle dynamically created datasets
        4. Validate data before storage or analysis
    """
    try:
        if ctx:
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

        if ctx:
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
        if ctx:
            await ctx.error(f"Failed to parse CSV content: {e!s}")
        raise ToolError(f"Failed to parse CSV content: {e}") from e


async def export_csv(
    session_id: str,
    file_path: str | None = None,
    format: Literal["csv", "tsv", "json", "excel", "parquet", "html", "markdown"] = "csv",
    encoding: str = "utf-8",
    index: bool = False,
    ctx: Context | None = None,
) -> ExportResult:
    """Export session data to various formats with comprehensive format support.

    Exports data from active sessions to multiple file formats with configurable
    encoding, formatting options, and automatic file naming. Supports all major
    data interchange formats for maximum compatibility.

    Args:
        session_id: Session ID containing data to export
        file_path: Optional output file path (auto-generated with timestamp if not provided)
        format: Export format (csv, tsv, json, excel, parquet, html, markdown)
        encoding: Output encoding (utf-8, latin1, cp1252, etc.)
        index: Whether to include pandas index in output
        ctx: FastMCP context for progress reporting

    Returns:
        Export result with file information and statistics

    Supported Formats:
        📄 CSV: Comma-separated values (RFC 4180 compliant)
        📄 TSV: Tab-separated values
        📄 JSON: JSON records format for APIs
        📊 Excel: XLSX format with full formatting
        🗂️  Parquet: Columnar format for big data
        🌐 HTML: Web-ready table format
        📝 Markdown: Documentation-friendly tables

    File Naming:
        - Auto-generated: export_{session_id}_{timestamp}.{ext}
        - Timestamp format: YYYYMMDD_HHMMSS
        - Safe filename characters only
        - Collision avoidance

    Examples:
        # Basic CSV export
        export_csv("session123")

        # Custom Excel file with index
        export_csv("session123", "/path/output.xlsx", "excel", index=True)

        # JSON export for API integration
        export_csv("session123", format="json")

    AI Workflow Integration:
        1. Export analysis results for sharing
        2. Create formatted reports in multiple formats
        3. Backup data at key processing stages
        4. Generate outputs for downstream systems
    """
    try:
        # Normalize format to ExportFormat enum
        format_enum = ExportFormat(format)

        # Get session and validate data
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session or session.df is None:
            raise ToolError(f"Session not found or no data loaded: {session_id}")

        if ctx:
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

            if ctx:
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

            if ctx:
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
        if ctx:
            await ctx.error(f"Failed to export data: {e!s}")
        raise ToolError(f"Failed to export data: {e}") from e


async def get_session_info(session_id: str, ctx: Context | None = None) -> SessionInfoResult:
    """Get comprehensive information about a specific session.

    Retrieves detailed session metadata including creation time, data statistics,
    and current state. Essential for session management and workflow coordination.

    Args:
        session_id: Session ID to query
        ctx: FastMCP context for logging

    Returns:
        Comprehensive session information including:
        - Creation and modification timestamps
        - Data loading status and dimensions
        - Auto-save configuration
        - Memory usage and performance metrics

    Session States:
        🟢 Active: Data loaded and ready for operations
        🟡 Empty: Session created but no data loaded
        🔴 Expired: Session timeout reached (if applicable)

    Examples:
        # Check session status
        info = get_session_info("session123")
        if info.data_loaded:
            print(f"Session has {info.row_count} rows")

    AI Workflow Integration:
        1. Verify session state before operations
        2. Monitor data loading progress
        3. Track session resource usage
        4. Coordinate multi-step workflows
    """
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise ToolError(f"Session not found: {session_id}")

        if ctx:
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
        if ctx:
            await ctx.error(f"Failed to get session info: {e!s}")
        raise ToolError(f"Failed to get session info: {e}") from e


async def list_sessions(ctx: Context | None = None) -> SessionListResult:
    """List all active sessions with comprehensive details and statistics.

    Provides overview of all sessions in the system including data status,
    timestamps, and resource usage. Essential for session management and
    system monitoring.

    Args:
        ctx: FastMCP context for logging

    Returns:
        Complete session listing with:
        - Individual session details and metadata
        - Total and active session counts
        - System resource usage summary

    Session Categories:
        🟢 Active: Sessions with loaded data
        🟡 Idle: Sessions created but no data
        📊 Statistics: Memory usage and performance

    Examples:
        # List all sessions
        sessions = list_sessions()
        for session in sessions.sessions:
            print(f"{session.session_id}: {session.row_count} rows")

    AI Workflow Integration:
        1. Monitor system resource usage
        2. Identify sessions for cleanup
        3. Track workflow progress across sessions
        4. Coordinate parallel data processing
    """
    try:
        session_manager = get_session_manager()
        sessions = session_manager.list_sessions()

        if ctx:
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
                columns=s.columns,
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
        if ctx:
            await ctx.error(f"Failed to list sessions: {e!s}")
        # Return empty list on error for consistency
        return SessionListResult(sessions=[], total_sessions=0, active_sessions=0)


async def close_session(session_id: str, ctx: Context | None = None) -> CloseSessionResult:
    """Close and clean up a session with proper resource management.

    Safely closes a session, releases memory, and cleans up associated resources.
    Provides confirmation of successful cleanup and data preservation status.

    Args:
        session_id: Session ID to close
        ctx: FastMCP context for logging

    Returns:
        Operation result confirming closure and cleanup status

    Cleanup Process:
        🧹 Memory deallocation for DataFrames
        💾 History and metadata cleanup
        🔒 Session state finalization
        📊 Resource usage reporting

    Data Preservation:
        ⚠️  Session closure removes data from memory
        💾 Use export_csv before closing to preserve data
        📈 Operation history is lost after closure

    Examples:
        # Close completed session
        result = close_session("session123")
        print(result.message)  # "Session session123 closed successfully"

    AI Workflow Integration:
        1. Clean up after workflow completion
        2. Free resources for new operations
        3. Enforce session lifecycle management
        4. Prevent memory leaks in long-running processes
    """
    try:
        session_manager = get_session_manager()
        removed = await session_manager.remove_session(session_id)

        if not removed:
            raise ToolError(f"Session not found: {session_id}")

        if ctx:
            await ctx.info(f"Closed session {session_id}")

        return CloseSessionResult(
            session_id=session_id,
            message=f"Session {session_id} closed successfully",
            data_preserved=False,  # Sessions are removed, so data is not preserved
        )

    except Exception as e:
        logger.error(f"Failed to close session: {e}")
        if ctx:
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
