"""FastMCP I/O tool definitions for CSV Editor."""

from __future__ import annotations

from typing import Any

from fastmcp import Context  # noqa: TC002

from ..models import ExportFormat
from .io_operations import (
    close_session as _close_session,
)
from .io_operations import (
    export_csv as _export_csv,
)
from .io_operations import (
    get_session_info as _get_session_info,
)
from .io_operations import (
    list_sessions as _list_sessions,
)
from .io_operations import (
    load_csv as _load_csv,
)
from .io_operations import (
    load_csv_from_content as _load_csv_from_content,
)
from .io_operations import (
    load_csv_from_url as _load_csv_from_url,
)


def register_io_tools(mcp: Any) -> None:
    """Register I/O tools with FastMCP server."""

    @mcp.tool
    async def load_csv(
        file_path: str,
        encoding: str = "utf-8",
        delimiter: str = ",",
        session_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Load a CSV file into a session with robust parsing and AI-optimized data preview.

        Provides comprehensive CSV loading with support for various encodings, delimiters, and
        complex quoted data including commas and escaped quotes. Optimized for AI workflows
        with enhanced data preview and coordinate system information.

        Args:
            file_path: Path to the CSV file (absolute or relative)
            encoding: File encoding (default: utf-8, also supports: latin1, cp1252, etc.)
            delimiter: Column delimiter (default: comma, also supports: tab, semicolon, pipe)
            session_id: Optional existing session ID, or None to create new session
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
            2. Use get_data_summary(session_id) for overview
            3. Inspect with get_cell_value/get_row_data for details
            4. Apply transformations based on data understanding
        """
        return await _load_csv(file_path, encoding, delimiter, session_id, ctx=ctx)

    @mcp.tool
    async def load_csv_from_url(
        url: str,
        encoding: str = "utf-8",
        delimiter: str = ",",
        session_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Load a CSV file from a URL."""
        return await _load_csv_from_url(url, encoding, delimiter, session_id, ctx)

    @mcp.tool
    async def load_csv_from_content(
        content: str,
        delimiter: str = ",",
        session_id: str | None = None,
        has_header: bool = True,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Load CSV data from string content."""
        return await _load_csv_from_content(content, delimiter, session_id, has_header, ctx)

    @mcp.tool
    async def export_csv(
        session_id: str,
        file_path: str | None = None,
        format: str = "csv",
        encoding: str = "utf-8",
        index: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Export session data to various formats."""
        format_enum = ExportFormat(format)
        return await _export_csv(session_id, file_path, format_enum, encoding, index, ctx)

    @mcp.tool
    async def get_session_info(session_id: str, ctx: Context | None = None) -> dict[str, Any]:
        """Get information about a specific session."""
        return await _get_session_info(session_id, ctx)

    @mcp.tool
    async def list_sessions(ctx: Context | None = None) -> dict[str, Any]:
        """List all active sessions."""
        return await _list_sessions(ctx)

    @mcp.tool
    async def close_session(session_id: str, ctx: Context | None = None) -> dict[str, Any]:
        """Close and clean up a session."""
        return await _close_session(session_id, ctx)
