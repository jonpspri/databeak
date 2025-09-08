"""I/O operations tools for CSV Editor MCP Server."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from fastmcp.exceptions import ToolError

from ..models import ExportFormat, OperationType, get_session_manager
from ..models.tool_responses import (
    CloseSessionResult,
    DataPreview,
    ExportResult,
    LoadResult,
    SessionInfoResult,
    SessionListResult,
)
from ..models.tool_responses import (
    SessionInfo as SessionInfoResponse,
)
from ..utils.validators import validate_file_path, validate_url
from .data_operations import create_data_preview_with_indices

if TYPE_CHECKING:
    from fastmcp import Context


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
    """Load a CSV file into a session.

    Args:
        file_path: Path to the CSV file
        encoding: File encoding (default: utf-8)
        delimiter: Column delimiter (default: comma)
        session_id: Optional existing session ID to use
        header: Row number to use as header (default: 0)
        na_values: Additional strings to recognize as NA/NaN
        parse_dates: Columns to parse as dates
        ctx: FastMCP context

    Returns:
        Operation result with session ID and data info
    """
    try:
        # Validate file path
        is_valid, validated_path = validate_file_path(file_path)
        if not is_valid:
            raise ToolError(f"Invalid file path: {validated_path}")

        if ctx:
            await ctx.info(f"Loading CSV file: {validated_path}")
            await ctx.report_progress(0.1)

        # Get or create session
        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(session_id)

        if ctx:
            await ctx.report_progress(0.3)

        # Read CSV with pandas
        read_params: dict[str, Any] = {
            "filepath_or_buffer": validated_path,
            "encoding": encoding,
            "delimiter": delimiter,
            "header": header,
        }

        if na_values:
            read_params["na_values"] = na_values
        if parse_dates:
            read_params["parse_dates"] = parse_dates

        df = pd.read_csv(**read_params)

        if ctx:
            await ctx.report_progress(0.8)

        # Load into session
        session.load_data(df, validated_path)

        if ctx:
            await ctx.report_progress(1.0)
            await ctx.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

        # Create data preview
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
            columns_affected=df.columns.tolist(),
            data=data_preview,
            memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
        )

    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to load CSV: {e!s}")
        raise ToolError(f"Failed to load CSV: {e}") from e


async def load_csv_from_url(
    url: str,
    encoding: str = "utf-8",
    delimiter: str = ",",
    session_id: str | None = None,
    ctx: Context | None = None,
) -> LoadResult:
    """Load a CSV file from a URL.

    Args:
        url: URL of the CSV file
        encoding: File encoding
        delimiter: Column delimiter
        session_id: Optional existing session ID
        ctx: FastMCP context

    Returns:
        Operation result with session ID and data info
    """
    try:
        # Validate URL
        is_valid, validated_url = validate_url(url)
        if not is_valid:
            raise ToolError(f"Invalid URL: {validated_url}")

        if ctx:
            await ctx.info(f"Loading CSV from URL: {url}")
            await ctx.report_progress(0.1)

        # Download CSV using pandas (it handles URLs directly)
        df = pd.read_csv(url, encoding=encoding, delimiter=delimiter)

        if ctx:
            await ctx.report_progress(0.8)

        # Get or create session
        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(session_id)
        session.load_data(df, url)

        if ctx:
            await ctx.report_progress(1.0)
            await ctx.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

        # Create data preview
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
            columns_affected=df.columns.tolist(),
            data=data_preview,
        )

    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to load CSV from URL: {e!s}")
        raise ToolError(f"Failed to load CSV from URL: {e}") from e


async def load_csv_from_content(
    content: str,
    delimiter: str = ",",
    session_id: str | None = None,
    has_header: bool = True,
    ctx: Context | None = None,
) -> LoadResult:
    """Load CSV data from a string content.

    Args:
        content: CSV content as string
        delimiter: Column delimiter
        session_id: Optional existing session ID
        has_header: Whether first row is header
        ctx: FastMCP context

    Returns:
        Operation result with session ID and data info
    """
    try:
        if ctx:
            await ctx.info("Loading CSV from content string")

        # Parse CSV from string
        from io import StringIO

        df = pd.read_csv(StringIO(content), delimiter=delimiter, header=0 if has_header else None)

        # Get or create session
        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(session_id)
        session.load_data(df, None)

        if ctx:
            await ctx.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

        # Create data preview
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
            columns_affected=df.columns.tolist(),
            data=data_preview,
        )

    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to parse CSV content: {e!s}")
        raise ToolError(f"Failed to parse CSV content: {e}") from e


async def export_csv(
    session_id: str,
    file_path: str | None = None,
    format: ExportFormat | str = ExportFormat.CSV,
    encoding: str = "utf-8",
    index: bool = False,
    ctx: Context | None = None,
) -> ExportResult:
    """Export session data to various formats.

    Args:
        session_id: Session ID to export
        file_path: Optional output file path (auto-generated if not provided)
        format: Export format (csv, tsv, json, excel, parquet, html, markdown)
        encoding: Output encoding
        index: Whether to include index in output
        ctx: FastMCP context

    Returns:
        Operation result with file path
    """
    try:
        # Normalize format to ExportFormat enum
        if isinstance(format, str):
            format_enum = ExportFormat(format)
        else:
            format_enum = format

        # Get session
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session or session.data_session.df is None:
            raise ToolError(f"Session not found or no data loaded: {session_id}")

        if ctx:
            await ctx.info(f"Exporting data in {format_enum} format")
            await ctx.report_progress(0.1)

        # Generate file path if not provided
        if not file_path:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"export_{session_id[:8]}_{timestamp}"

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

            file_path = tempfile.gettempdir() + "/" + filename + extensions[format_enum]

        path_obj = Path(file_path)
        df = session.data_session.df

        if ctx:
            await ctx.report_progress(0.5)

        # Export based on format
        if format_enum == ExportFormat.CSV:
            df.to_csv(path_obj, encoding=encoding, index=index)
        elif format_enum == ExportFormat.TSV:
            df.to_csv(path_obj, sep="\t", encoding=encoding, index=index)
        elif format_enum == ExportFormat.JSON:
            df.to_json(path_obj, orient="records", indent=2)
        elif format_enum == ExportFormat.EXCEL:
            df.to_excel(path_obj, index=index, engine="openpyxl")
        elif format_enum == ExportFormat.PARQUET:
            df.to_parquet(path_obj, index=index)
        elif format_enum == ExportFormat.HTML:
            df.to_html(path_obj, index=index)
        elif format_enum == ExportFormat.MARKDOWN:
            df.to_markdown(path_obj, index=index)
        else:
            raise ToolError(f"Unsupported format: {format_enum}")

        # Record operation
        session.record_operation(
            OperationType.EXPORT, {"format": format_enum, "file_path": str(file_path)}
        )

        if ctx:
            await ctx.report_progress(1.0)
            await ctx.info(f"Exported to {file_path}")

        return ExportResult(
            session_id=session_id,
            file_path=str(file_path),
            format=format_enum.value,
            rows_exported=len(df),
            file_size_mb=path_obj.stat().st_size / (1024 * 1024),
        )

    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to export data: {e!s}")
        raise ToolError(f"Failed to export data: {e}") from e


async def get_session_info(session_id: str, ctx: Context | None = None) -> SessionInfoResult:
    """Get information about a specific session.

    Args:
        session_id: Session ID
        ctx: FastMCP context

    Returns:
        Session information
    """
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise ToolError(f"Session not found: {session_id}")

        if ctx:
            await ctx.info(f"Retrieved info for session {session_id}")

        info = session.get_info()
        return SessionInfoResult(
            session_id=session_id,
            created_at=info.created_at.isoformat(),
            last_modified=info.last_accessed.isoformat(),
            data_loaded=session.data_session.df is not None,
            row_count=info.row_count if session.data_session.df is not None else None,
            column_count=info.column_count if session.data_session.df is not None else None,
            auto_save_enabled=session.auto_save_config.enabled,
        )

    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to get session info: {e!s}")
        raise ToolError(f"Failed to get session info: {e}") from e


async def list_sessions(ctx: Context | None = None) -> SessionListResult:
    """List all active sessions.

    Args:
        ctx: FastMCP context

    Returns:
        List of active sessions
    """
    try:
        session_manager = get_session_manager()
        sessions = session_manager.list_sessions()

        if ctx:
            await ctx.info(f"Found {len(sessions)} active sessions")

        # Convert session info to SessionInfoResponse objects for tool response
        session_infos = []
        active_count = 0
        for s in sessions:
            # s is already a SessionInfo from data_models, so we can access its attributes directly
            session_info = SessionInfoResponse(
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
        if ctx:
            await ctx.error(f"Failed to list sessions: {e!s}")
        # Return empty list on error for consistency
        return SessionListResult(sessions=[], total_sessions=0, active_sessions=0)


async def close_session(session_id: str, ctx: Context | None = None) -> CloseSessionResult:
    """Close and clean up a session.

    Args:
        session_id: Session ID to close
        ctx: FastMCP context

    Returns:
        Operation result
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
        if ctx:
            await ctx.error(f"Failed to close session: {e!s}")
        raise ToolError(f"Failed to close session: {e}") from e
