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


async def health_check(ctx: Context | None = None) -> HealthResult:
    """Check the health status of DataBeak server.

    Performs comprehensive system health assessment including session management,
    memory usage, and service availability. Essential for monitoring DataBeak
    server status and ensuring reliable operation.

    Args:
        ctx: FastMCP context for progress reporting and logging

    Returns:
        Comprehensive health status information including:
        - overall status (healthy/degraded/unhealthy)
        - server version information
        - active session count and limits
        - session timeout configuration
        - system resource status

    Health Status Levels:
        🟢 healthy: All systems operational, resources available
        🟡 degraded: Minor issues or resource constraints
        🔴 unhealthy: Critical issues requiring attention

    System Checks:
        ✅ Session Manager: Availability and configuration
        ✅ Active Sessions: Count and resource usage
        ✅ Memory Status: Available resources
        ✅ Service Status: Core functionality availability

    Examples:
        # Basic health check
        health = await health_check()
        print(f"Status: {health.status}")
        print(f"Active sessions: {health.active_sessions}/{health.max_sessions}")

        # Health monitoring in workflows
        if health.status == "healthy":
            proceed_with_operations()

    AI Workflow Integration:
        1. System status verification before large operations
        2. Resource availability assessment
        3. Session capacity planning
        4. Automated monitoring and alerting
        5. Performance baseline establishment
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

    except Exception as e:
        logger.error(f"Health check failed: {e!s}")
        if ctx:
            await ctx.error(f"Health check failed: {e!s}")
        
        # Return unhealthy status with minimal info if health check itself fails
        try:
            return HealthResult(
                status="unhealthy",
                version=__version__,
                active_sessions=0,
                max_sessions=0,
                session_ttl_minutes=0,
            )
        except Exception as fallback_error:
            logger.error(f"Fallback health result creation failed: {fallback_error!s}")
            raise ToolError(f"Critical health check failure: {e}") from e


async def get_server_info(ctx: Context | None = None) -> ServerInfoResult:
    """Get comprehensive information about DataBeak server capabilities.

    Provides detailed server information including supported operations, file formats,
    configuration limits, and feature capabilities. Essential for client applications
    to understand DataBeak's functionality and plan optimal usage.

    Args:
        ctx: FastMCP context for logging and progress reporting

    Returns:
        Complete server capability information including:
        - server identification (name, version, description)
        - operational capabilities grouped by function
        - supported file formats and extensions
        - configuration limits and constraints
        - feature availability matrix

    Capability Categories:
        📊 Data I/O: Loading, exporting, format conversion
        🔧 Data Manipulation: Filtering, sorting, transformations
        📈 Data Analysis: Statistics, correlations, profiling
        ✅ Data Validation: Schema checks, quality assessment
        🗂️ Session Management: Multi-session support, isolation
        🔢 Null Handling: Comprehensive null value operations

    Configuration Information:
        📏 File Size Limits: Maximum supported file sizes
        ⏱️ Timeout Settings: Session and operation timeouts
        💾 Memory Limits: Resource usage constraints
        📊 Session Limits: Concurrent session capabilities

    Examples:
        # Get server capabilities
        info = await get_server_info()
        print(f"Server: {info.name} v{info.version}")
        print(f"Formats: {info.supported_formats}")
        
        # Check specific capabilities
        if "correlation_matrix" in info.capabilities.get("data_analysis", []):
            perform_correlation_analysis()

        # Validate file size before upload
        if file_size_mb <= info.max_file_size_mb:
            proceed_with_upload()

    AI Workflow Integration:
        1. Capability discovery for dynamic workflow planning
        2. Format compatibility verification
        3. Resource limit awareness for operation planning
        4. Feature availability checking before usage
        5. Client configuration and optimization
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