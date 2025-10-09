"""Standalone System server for DataBeak using FastMCP server composition.

This module provides a complete System server implementation following DataBeak's modular server
architecture pattern. It includes health monitoring, server capability information, and system
status reporting with comprehensive error handling and AI-optimized responses.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated

import psutil
from fastmcp import Context, FastMCP
from pydantic import Field

# Import version and session management from main package
from databeak._version import __version__
from databeak.core.session import get_session_manager
from databeak.core.settings import DataBeakSettings, get_settings
from databeak.models.tool_responses import HealthResult, ServerInfoResult

logger = logging.getLogger(__name__)

# ============================================================================
# MEMORY MONITORING CONSTANTS AND UTILITIES
# ============================================================================

# Memory monitoring will use configurable thresholds from DataBeakSettings


def get_memory_usage() -> float:
    """Get current process memory usage in MB.

    Returns:
        Memory usage in MB, or 0.0 if measurement fails

    Raises:
        None - errors are logged and 0.0 is returned for resilience

    """
    try:
        process = psutil.Process(os.getpid())
        memory_bytes: int = process.memory_info().rss
        return float(memory_bytes / 1024 / 1024)
    except (psutil.Error, OSError) as e:
        logger.warning("Failed to get memory usage: %s", str(e))
        return 0.0


def get_memory_status(
    current_mb: float, threshold_mb: float, settings: DataBeakSettings | None = None
) -> str:
    """Determine memory status based on configurable thresholds.

    Returns:
        Status string: 'normal', 'warning', or 'critical'

    """
    if settings is None:
        settings = get_settings()

    usage_ratio = current_mb / threshold_mb if threshold_mb > 0 else 0

    if usage_ratio >= settings.memory_critical_threshold:
        return "critical"
    if usage_ratio >= settings.memory_warning_threshold:
        return "warning"
    return "normal"


# ============================================================================
# SYSTEM OPERATIONS LOGIC - DIRECT IMPLEMENTATIONS
# ============================================================================


# Health check implementation details:
# - Performs comprehensive system assessment including session management
# - Checks memory usage and service availability
# - Status levels: healthy (operational), degraded (constraints), unhealthy (critical issues)
# - System checks: Session Manager availability, Active Sessions count, Memory Status, Service Status
async def health_check(
    ctx: Annotated[Context, Field(description="FastMCP context for progress reporting")],
) -> HealthResult:
    """Check DataBeak server health and availability with memory monitoring.

    Returns server status, session capacity, memory usage, and version information. Use before large
    operations to verify system readiness and resource availability.
    """
    try:
        await ctx.info("Performing DataBeak health check with memory monitoring")

        session_manager = get_session_manager()
        settings = get_settings()
        active_sessions = len(session_manager.sessions)

        # Get memory information
        current_memory_mb = get_memory_usage()
        health_threshold_mb = float(settings.health_memory_threshold_mb)
        memory_status = get_memory_status(current_memory_mb, health_threshold_mb)

        # Determine overall health status
        status = "healthy"

        # Check session capacity using configurable threshold
        session_capacity_ratio = active_sessions / session_manager.max_sessions
        if session_capacity_ratio >= settings.session_capacity_warning_threshold:
            status = "degraded"
            await ctx.warning(
                f"Session capacity warning: {active_sessions}/{session_manager.max_sessions} "
                f"({session_capacity_ratio:.1%})"
            )

        # Check memory status
        if memory_status == "critical":
            status = "unhealthy"
            await ctx.error(
                f"Critical memory usage: {current_memory_mb:.1f}MB / {health_threshold_mb:.1f}MB"
            )
        elif memory_status == "warning":
            if status == "healthy":
                status = "degraded"
            await ctx.warning(
                f"High memory usage: {current_memory_mb:.1f}MB / {health_threshold_mb:.1f}MB"
            )

        await ctx.info(
            f"Health check complete - Status: {status}, Sessions: {active_sessions}, "
            f"Memory: {current_memory_mb:.1f}MB ({memory_status})"
        )

        return HealthResult(
            status=status,
            version=__version__,
            active_sessions=active_sessions,
            max_sessions=session_manager.max_sessions,
            session_ttl_minutes=session_manager.ttl_minutes,
            memory_usage_mb=current_memory_mb,
            memory_threshold_mb=health_threshold_mb,
            memory_status=memory_status,
            history_operations_total=0,  # History operations tracking removed
            history_limit_per_session=0,  # History operations tracking removed
        )

    except (ImportError, AttributeError, ValueError, TypeError) as e:
        # Handle specific configuration/import issues - return unhealthy
        await ctx.error(f"Health check failed due to configuration issue: {e}")

        return HealthResult(
            status="unhealthy",
            version="unknown",
            active_sessions=0,
            max_sessions=0,
            session_ttl_minutes=0,
            memory_usage_mb=0.0,
            memory_threshold_mb=2048.0,
            memory_status="unknown",
            history_operations_total=0,  # History operations tracking removed
            history_limit_per_session=0,  # History operations tracking removed
        )
    except Exception as e:
        # Treat unexpected session manager errors as recoverable - return unhealthy
        await ctx.error(f"Health check failed: {e}")

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
            memory_usage_mb=0.0,
            memory_threshold_mb=2048.0,
            memory_status="unknown",
            history_operations_total=0,  # History operations tracking removed
            history_limit_per_session=0,  # History operations tracking removed
        )


# Server info implementation details:
# - Capability categories: Data I/O, Manipulation, Analysis, Validation, Session Management, Null Handling
# - Configuration info: File size limits, timeout settings, memory limits, session limits
# - Used for capability discovery, format compatibility verification, resource limit awareness
async def get_server_info(
    ctx: Annotated[Context, Field(description="FastMCP context for progress reporting")],
) -> ServerInfoResult:
    """Get DataBeak server capabilities and supported operations.

    Returns server version, available tools, supported file formats, and resource limits. Use to
    discover what operations are available before planning workflows.
    """
    await ctx.info("Retrieving DataBeak server information")

    # Get current configuration settings
    settings = get_settings()

    server_info = ServerInfoResult(
        name="DataBeak",
        version=__version__,
        description="A comprehensive MCP server for CSV file operations and data analysis",
        capabilities={
            "data_io": [
                "load_csv_from_url",
                "load_csv_from_content",
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
        max_download_size_mb=settings.max_download_size_mb,
        session_timeout_minutes=settings.session_timeout // 60,
    )

    await ctx.info("Server information retrieved successfully")

    return server_info


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
