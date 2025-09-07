"""FastMCP I/O tool definitions for DataBeak."""

from __future__ import annotations

from typing import Any

from fastmcp import Context, FastMCP
from pydantic import BaseModel

from ..models import ExportFormat
from .io_operations import close_session as _close_session
from .io_operations import export_csv as _export_csv
from .io_operations import get_session_info as _get_session_info
from .io_operations import list_sessions as _list_sessions
from .io_operations import load_csv as _load_csv
from .io_operations import load_csv_from_content as _load_csv_from_content
from .io_operations import load_csv_from_url as _load_csv_from_url


class LoadResult(BaseModel):
    """Response model for data loading operations."""

    success: bool = True
    session_id: str
    rows_affected: int
    columns_affected: list[str]
    data: dict[str, Any] | None = None
    memory_usage_mb: float | None = None


class ExportResult(BaseModel):
    """Response model for data export operations."""

    success: bool = True
    session_id: str
    file_path: str
    format: str
    rows_exported: int
    file_size_mb: float | None = None


class SessionInfoResult(BaseModel):
    """Response model for session information."""

    success: bool = True
    session_id: str
    created_at: str
    last_modified: str
    data_loaded: bool
    row_count: int | None = None
    column_count: int | None = None
    auto_save_enabled: bool


class SessionListResult(BaseModel):
    """Response model for listing all sessions."""

    success: bool = True
    sessions: list[dict[str, Any]]
    total_sessions: int
    active_sessions: int


class CloseSessionResult(BaseModel):
    """Response model for session closure operations."""

    success: bool = True
    session_id: str
    message: str
    data_preserved: bool


def register_io_tools(mcp: FastMCP) -> None:
    """Register I/O tools with FastMCP server."""

    @mcp.tool
    async def load_csv(
        file_path: str,
        encoding: str = "utf-8",
        delimiter: str = ",",
        session_id: str | None = None,
        ctx: Context | None = None,
    ) -> LoadResult:
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
        result = await _load_csv(file_path, encoding, delimiter, session_id, ctx=ctx)

        # Convert dict response to Pydantic model
        return LoadResult(
            session_id=result.get("session_id", ""),
            rows_affected=result.get("rows_affected", 0),
            columns_affected=result.get("columns_affected", []),
            data=result.get("data"),
            memory_usage_mb=result.get("memory_usage_mb"),
        )

    @mcp.tool
    async def load_csv_from_url(
        url: str,
        encoding: str = "utf-8",
        delimiter: str = ",",
        session_id: str | None = None,
        ctx: Context | None = None,
    ) -> LoadResult:
        """Load a CSV file from a URL."""
        result = await _load_csv_from_url(url, encoding, delimiter, session_id, ctx)

        # Convert dict response to Pydantic model
        return LoadResult(
            session_id=result.get("session_id", ""),
            rows_affected=result.get("rows_affected", 0),
            columns_affected=result.get("columns_affected", []),
            data=result.get("data"),
            memory_usage_mb=result.get("memory_usage_mb"),
        )

    @mcp.tool
    async def load_csv_from_content(
        content: str,
        delimiter: str = ",",
        session_id: str | None = None,
        has_header: bool = True,
        ctx: Context | None = None,
    ) -> LoadResult:
        """Load CSV data from string content."""
        result = await _load_csv_from_content(content, delimiter, session_id, has_header, ctx)

        # Convert dict response to Pydantic model
        return LoadResult(
            session_id=result.get("session_id", ""),
            rows_affected=result.get("rows_affected", 0),
            columns_affected=result.get("columns_affected", []),
            data=result.get("data"),
            memory_usage_mb=result.get("memory_usage_mb"),
        )

    @mcp.tool
    async def export_csv(
        session_id: str,
        file_path: str | None = None,
        format: str = "csv",
        encoding: str = "utf-8",
        index: bool = False,
        ctx: Context | None = None,
    ) -> ExportResult:
        """Export session data to various formats."""
        format_enum = ExportFormat(format)
        result = await _export_csv(session_id, file_path, format_enum, encoding, index, ctx)

        # Convert dict response to Pydantic model
        return ExportResult(
            session_id=session_id,
            file_path=result.get("file_path", file_path or ""),
            format=format,
            rows_exported=result.get("rows_exported", 0),
            file_size_mb=result.get("file_size_mb"),
        )

    @mcp.tool
    async def get_session_info(session_id: str, ctx: Context | None = None) -> SessionInfoResult:
        """Get information about a specific session."""
        result = await _get_session_info(session_id, ctx)

        # Convert dict response to Pydantic model
        return SessionInfoResult(
            session_id=session_id,
            created_at=result.get("created_at", ""),
            last_modified=result.get("last_modified", ""),
            data_loaded=result.get("data_loaded", False),
            row_count=result.get("row_count"),
            column_count=result.get("column_count"),
            auto_save_enabled=result.get("auto_save_enabled", False),
        )

    @mcp.tool
    async def list_sessions(ctx: Context | None = None) -> SessionListResult:
        """List all active sessions."""
        result = await _list_sessions(ctx)

        # Convert dict response to Pydantic model
        sessions_data = result.get("sessions", [])
        return SessionListResult(
            sessions=sessions_data,
            total_sessions=len(sessions_data),
            active_sessions=result.get("active_sessions", len(sessions_data)),
        )

    @mcp.tool
    async def close_session(session_id: str, ctx: Context | None = None) -> CloseSessionResult:
        """Close and clean up a session."""
        result = await _close_session(session_id, ctx)

        # Convert dict response to Pydantic model
        return CloseSessionResult(
            session_id=session_id,
            message=result.get("message", "Session closed successfully"),
            data_preserved=result.get("data_preserved", False),
        )
