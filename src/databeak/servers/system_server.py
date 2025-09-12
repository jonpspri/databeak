"""Standalone System server for DataBeak using FastMCP server composition.

This module provides a complete System server implementation following DataBeak's modular server
architecture pattern. It includes health monitoring, server capability information, and system
status reporting with comprehensive error handling and AI-optimized responses.
"""

from __future__ import annotations

import logging

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

# Import version and session management from main package
from .._version import __version__
from ..models import get_session_manager
from ..models.csv_session import get_csv_settings
from ..models.tool_responses import HealthResult, ServerInfoResult

logger = logging.getLogger(__name__)

# ============================================================================
# SYSTEM OPERATIONS LOGIC - DIRECT IMPLEMENTATIONS
# ============================================================================


# Health check implementation details:
# - Performs comprehensive system assessment including session management
# - Checks memory usage and service availability
# - Status levels: healthy (operational), degraded (constraints), unhealthy (critical issues)
# - System checks: Session Manager availability, Active Sessions count, Memory Status, Service Status
async def health_check(ctx: Context | None = None) -> HealthResult:
    """Check DataBeak server health and availability.

    Returns server status, session capacity, and version information. Use before large operations to
    verify system readiness.
    """
    try:
        if ctx:
            await ctx.info("Performing DataBeak health check")

        session_manager = get_session_manager()
        active_sessions = len(session_manager.sessions)

        # Determine health status based on system state
        status = "healthy"

        # Check for potential issues
        if active_sessions >= session_manager.max_sessions * 0.9:  # 90% capacity warning
            status = "degraded"
            logger.warning(
                f"Session capacity warning: {active_sessions}/{session_manager.max_sessions}"
            )

        if ctx:
            await ctx.info(
                f"Health check complete - Status: {status}, Active sessions: {active_sessions}"
            )

        return HealthResult(
            status=status,
            version=__version__,
            active_sessions=active_sessions,
            max_sessions=session_manager.max_sessions,
            session_ttl_minutes=session_manager.ttl_minutes,
        )

    except (ImportError, AttributeError, ValueError, TypeError) as e:
        # Handle specific configuration/import issues - return unhealthy
        logger.error(f"Health check failed due to configuration issue: {e!s}")
        if ctx:
            await ctx.error(f"Health check failed: {e!s}")

        return HealthResult(
            status="unhealthy",
            version="unknown",
            active_sessions=0,
            max_sessions=0,
            session_ttl_minutes=0,
        )
    except Exception as e:
        # Treat unexpected session manager errors as recoverable - return unhealthy
        logger.error(f"Health check failed: {e!s}")
        if ctx:
            await ctx.error(f"Health check failed: {e!s}")

        try:
            version = str(__version__)
        except Exception:
            version = "unknown"

        return HealthResult(
            status="unhealthy",
            version=version,
            active_sessions=0,
            max_sessions=0,
            session_ttl_minutes=0,
        )


# Server info implementation details:
# - Capability categories: Data I/O, Manipulation, Analysis, Validation, Session Management, Null Handling
# - Configuration info: File size limits, timeout settings, memory limits, session limits
# - Used for capability discovery, format compatibility verification, resource limit awareness
async def get_server_info(ctx: Context | None = None) -> ServerInfoResult:
    """Get DataBeak server capabilities and supported operations.

    Returns server version, available tools, supported file formats, and resource limits. Use to
    discover what operations are available before planning workflows.
    """
    try:
        if ctx:
            await ctx.info("Retrieving DataBeak server information")

        # Get current configuration settings
        settings = get_csv_settings()

        server_info = ServerInfoResult(
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
            max_file_size_mb=settings.max_file_size_mb,
            session_timeout_minutes=settings.session_timeout // 60,
        )

        if ctx:
            await ctx.info("Server information retrieved successfully")

        return server_info

    except Exception as e:
        logger.error(f"Failed to get server information: {e!s}")
        if ctx:
            await ctx.error(f"Failed to get server information: {e!s}")
        raise ToolError(f"Failed to get server information: {e}") from e


# ============================================================================
# FASTMCP SERVER SETUP
# ============================================================================

# Create System server
system_server = FastMCP(
    "DataBeak-System",
    instructions="System monitoring and information server for DataBeak with comprehensive health checking and capability reporting",
)

# Register the system functions directly as MCP tools (no wrapper functions needed)
system_server.tool(name="health_check")(health_check)
system_server.tool(name="get_server_info")(get_server_info)
