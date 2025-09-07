"""FastMCP system tool definitions for DataBeak."""

from __future__ import annotations

from fastmcp import Context, FastMCP
from pydantic import BaseModel

from .._version import __version__
from ..models import get_session_manager
from ..models.csv_session import get_csv_settings


class HealthResult(BaseModel):
    """Response model for system health check."""

    success: bool = True
    status: str
    version: str
    active_sessions: int
    max_sessions: int
    session_ttl_minutes: int


class ServerInfoResult(BaseModel):
    """Response model for server information and capabilities."""

    name: str
    version: str
    description: str
    capabilities: dict[str, list[str]]
    supported_formats: list[str]
    max_file_size_mb: int
    session_timeout_minutes: int


def register_system_tools(mcp: FastMCP) -> None:
    """Register system tools with FastMCP server."""

    @mcp.tool
    async def health_check(ctx: Context) -> HealthResult:
        """Check the health status of DataBeak."""
        session_manager = get_session_manager()

        active_sessions = len(session_manager.sessions)

        if ctx:
            await ctx.info("Health check performed successfully")

        return HealthResult(
            status="healthy",
            version=__version__,
            active_sessions=active_sessions,
            max_sessions=session_manager.max_sessions,
            session_ttl_minutes=session_manager.ttl_minutes,
        )

    @mcp.tool
    async def get_server_info(ctx: Context) -> ServerInfoResult:
        """Get information about DataBeak capabilities."""
        if ctx:
            await ctx.info("Server information requested")

        return ServerInfoResult(
            name="DataBeak",
            version=__version__,
            description="A comprehensive MCP server for CSV file operations and data analysis",
            capabilities={
                "data_io": [
                    "load_csv",
                    "load_csv_from_url",
                    "load_csv_from_content",
                    "export_csv",
                    "multiple_export_formats",
                ],
                "data_manipulation": [
                    "filter_rows",
                    "sort_data",
                    "select_columns",
                    "rename_columns",
                    "add_column",
                    "remove_columns",
                    "change_column_type",
                    "fill_missing_values",
                    "remove_duplicates",
                    "null_value_support",  # Explicitly mention null support
                ],
                "data_analysis": [
                    "get_statistics",
                    "correlation_matrix",
                    "group_by_aggregate",
                    "value_counts",
                    "detect_outliers",
                    "profile_data",
                ],
                "data_validation": [
                    "validate_schema",
                    "check_data_quality",
                    "find_anomalies",
                ],
                "session_management": [
                    "multi_session_support",
                    "session_isolation",
                    "auto_cleanup",
                ],
                "null_handling": [
                    "json_null_support",
                    "python_none_support",
                    "pandas_nan_compatibility",
                    "null_value_insertion",
                    "null_value_updates",
                ],
            },
            supported_formats=[
                "csv",
                "tsv",
                "json",
                "excel",
                "parquet",
                "html",
                "markdown",
            ],
            max_file_size_mb=get_csv_settings().max_file_size_mb,
            session_timeout_minutes=get_csv_settings().session_timeout // 60,
        )
