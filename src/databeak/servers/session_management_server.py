"""Unified Session Management server for DataBeak using FastMCP server composition.

This module consolidates both history management and auto-save functionality into a single
cohesive server, addressing the architectural issues where these closely-related session-level
concerns were split across multiple servers.

This consolidation resolves:
- Mixed responsibilities in history_server.py
- Duplicate AutoSaveConfig models
- Confusing separation of session-level operations
- Tight coupling between history and auto-save functionality
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Import session management from the main package
from ..exceptions import SessionNotFoundError
from ..models import get_session_manager
from ..models.typed_dicts import OperationDetails, SessionMetadataKeys


def safe_int(*, value: Any, default: int = 0) -> int:
    """Safely convert object to int with proper type constraints.

    Args:
        value: Value to convert (str, int, float, bool, or None) - keyword-only
        default: Default value if conversion fails - keyword-only

    Returns:
        Converted integer or default value
    """
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def safe_str(*, value: Any, default: str = "") -> str:
    """Safely convert object to str with proper type constraints.

    Args:
        value: Value to convert (str, int, float, bool, or None) - keyword-only
        default: Default value if conversion fails - keyword-only

    Returns:
        Converted string or default value
    """
    try:
        return str(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def safe_bool(*, value: Any, default: bool = False) -> bool:
    """Safely convert object to bool with proper type constraints.

    Args:
        value: Value to convert (str, int, float, bool, or None)
        default: Default value if conversion fails (keyword-only)

    Returns:
        Converted boolean or default value
    """
    try:
        return bool(value) if value is not None else default
    except (TypeError, ValueError):
        return default


logger = logging.getLogger(__name__)

# ============================================================================
# UNIFIED SESSION MANAGEMENT MODELS
# ============================================================================


class AutoSaveConfig(BaseModel):
    """Unified auto-save configuration model (replaces duplicate definitions)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(description="Whether auto-save is enabled")
    mode: Literal["disabled", "after_operation", "periodic", "hybrid"] = Field(
        description="Auto-save trigger mode",
    )
    strategy: Literal["overwrite", "backup", "versioned", "custom"] = Field(
        description="File saving strategy",
    )
    file_path: str | None = Field(
        default=None,
        description="Auto-save file path (None for default based on original file)",
    )
    format: Literal["csv", "tsv", "json", "excel", "parquet"] = Field(
        default="csv",
        description="Export format for auto-save",
    )
    interval_seconds: int = Field(
        default=300,
        description="Auto-save interval in seconds (for periodic mode)",
    )
    encoding: str = Field(default="utf-8", description="File encoding for auto-save")
    backup_count: int = Field(
        default=5,
        description="Number of backup files to keep (for backup/versioned strategies)",
    )

    @field_validator("interval_seconds")
    @classmethod
    def validate_interval(cls, v: int) -> int:
        """Validate auto-save interval."""
        if v < 10:
            msg = "Auto-save interval must be at least 10 seconds"
            raise ValueError(msg)
        return v

    @field_validator("backup_count")
    @classmethod
    def validate_backup_count(cls, v: int) -> int:
        """Validate backup count."""
        if v < 1:
            msg = "Backup count must be at least 1"
            raise ValueError(msg)
        return v


class AutoSaveStatus(BaseModel):
    """Auto-save status information."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(description="Whether auto-save is currently enabled")
    config: AutoSaveConfig | None = Field(
        default=None,
        description="Current auto-save configuration",
    )
    last_save_time: str | None = Field(
        default=None,
        description="Timestamp of last auto-save (ISO format)",
    )
    last_save_success: bool | None = Field(
        default=None,
        description="Whether the last auto-save was successful",
    )
    next_save_time: str | None = Field(
        default=None,
        description="Timestamp of next scheduled auto-save (ISO format)",
    )


class SessionOperationResult(BaseModel):
    """Unified response model for session management operations."""

    success: bool = Field(default=True, description="Whether operation completed successfully")
    message: str = Field(description="Operation result message")
    session_id: str = Field(description="Session ID")
    operation_type: str = Field(description="Type of operation performed")
    metadata: OperationDetails = Field(
        default_factory=dict, description="Structured operation metadata"
    )


# Response models for specific operations
class AutoSaveConfigResult(BaseModel):
    """Response model for auto-save configuration operations."""

    success: bool = Field(
        default=True, description="Whether the configuration update was successful"
    )
    config: AutoSaveConfig = Field(description="New auto-save configuration")
    previous_config: AutoSaveConfig | None = Field(
        None,
        description="Previous auto-save configuration",
    )


class AutoSaveStatusResult(BaseModel):
    """Response model for auto-save status operations."""

    success: bool = Field(default=True, description="Whether the status retrieval was successful")
    status: AutoSaveStatus = Field(description="Current auto-save status information")


class AutoSaveDisableResult(BaseModel):
    """Response model for auto-save disable operations."""

    success: bool = Field(default=True, description="Whether the disable operation was successful")
    was_enabled: bool = Field(
        default=False,
        description="Whether auto-save was enabled before disabling",
    )
    final_save_performed: bool = Field(
        default=False,
        description="Whether a final save was performed before disabling",
    )


class ManualSaveResult(BaseModel):
    """Response model for manual save operations."""

    success: bool = Field(default=True, description="Whether the manual save was successful")
    file_path: str = Field(description="Path where the file was saved")
    rows_saved: int = Field(description="Number of data rows saved")
    columns_saved: int = Field(description="Number of data columns saved")
    save_time: str = Field(description="Timestamp of save operation (ISO format)")


class HistoryOperationResult(BaseModel):
    """Response model for history operations (undo/redo)."""

    success: bool = Field(default=True, description="Whether the history operation was successful")
    message: str = Field(description="Operation result message")
    operation_details: dict[str, Any] | None = Field(
        default=None,
        description="Details of the operation that was undone/redone",
    )
    can_undo: bool = Field(description="Whether undo is currently possible")
    can_redo: bool = Field(description="Whether redo is currently possible")
    current_position: int = Field(description="Current position in operation history")


class HistoryListResult(BaseModel):
    """Response model for history listing operations."""

    success: bool = Field(default=True, description="Whether the history retrieval was successful")
    operations: list[dict[str, Any]] = Field(
        description="List of operations in chronological order"
    )
    total_operations: int = Field(description="Total number of operations in history")
    current_position: int = Field(description="Current position in operation history")
    can_undo: bool = Field(description="Whether undo is currently possible")
    can_redo: bool = Field(description="Whether redo is currently possible")


class HistoryExportResult(BaseModel):
    """Response model for history export operations."""

    success: bool = Field(default=True, description="Whether the history export was successful")
    session_id: str = Field(description="Session ID")
    created_at: str = Field(description="Session creation timestamp (ISO format)")
    operations: list[dict[str, Any]] = Field(description="Complete operation history")
    metadata: dict[str, Any] = Field(description="Session metadata")
    export_time: str = Field(description="Export timestamp (ISO format)")


# ============================================================================
# SESSION MANAGEMENT OPERATIONS - UNIFIED IMPLEMENTATION
# ============================================================================


async def undo_operation(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    session_id: Annotated[str, Field(description="Session identifier for the CSV data")],
) -> HistoryOperationResult:
    """Undo the last operation in the session history.

    Reverts the most recent data operation and restores the DataFrame to its previous state. Use
    when you need to reverse an unwanted operation or experiment with different approaches.
    """
    try:
        await ctx.info(f"Undoing last operation for session {session_id}")

        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise SessionNotFoundError(session_id)

        # Perform undo operation
        result = await session.undo()

        if result["success"]:
            await ctx.info(
                f"Successfully undid operation: {result.get('message', 'Unknown operation')}"
            )

            return HistoryOperationResult(
                success=True,
                message=result["message"],
                operation_details=result.get("operation"),
                can_undo=result.get("can_undo", False),
                can_redo=result.get("can_redo", False),
                current_position=0,  # Would need to track position in session
            )
        await ctx.warning(f"Undo failed: {result.get('error', 'Unknown error')}")

        return HistoryOperationResult(
            success=False,
            message=f"Undo failed: {result.get('error', 'Unknown error')}",
            operation_details=None,
            can_undo=False,
            can_redo=False,
            current_position=0,
        )

    except Exception as e:
        logger.error("Undo operation failed for session %s: %s", session_id, str(e))
        error_msg = f"Undo operation failed: {e}"
        await ctx.error(error_msg)
        raise ToolError(error_msg) from e


async def redo_operation(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    session_id: Annotated[str, Field(description="Session identifier for the CSV data")],
) -> HistoryOperationResult:
    """Redo a previously undone operation in the session history.

    Restores an operation that was previously undone using undo_operation. This allows you to
    experiment with undo/redo workflows and recover from accidental undo operations.
    """
    try:
        await ctx.info(f"Redoing last undone operation for session {session_id}")

        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise SessionNotFoundError(session_id)

        # Perform redo operation
        result = await session.redo()

        if result["success"]:
            await ctx.info(
                f"Successfully redid operation: {result.get('message', 'Unknown operation')}"
            )

            return HistoryOperationResult(
                success=True,
                message=result["message"],
                operation_details=result.get("operation"),
                can_undo=result.get("can_undo", False),
                can_redo=result.get("can_redo", False),
                current_position=0,
            )
        await ctx.warning(f"Redo failed: {result.get('error', 'Unknown error')}")

        return HistoryOperationResult(
            success=False,
            message=f"Redo failed: {result.get('error', 'Unknown error')}",
            operation_details=None,
            can_undo=False,
            can_redo=False,
            current_position=0,
        )

    except SessionNotFoundError as e:
        await ctx.error(f"Session not found: {e.details.get('session_id', session_id)}")
        msg = f"Session not found: {session_id}"
        raise ToolError(msg) from e
    except Exception as e:
        logger.error("Redo operation failed for session %s: %s", session_id, str(e))
        await ctx.error(f"Redo operation failed: {e}")
        msg = f"Redo operation failed: {e}"
        raise ToolError(msg) from e


async def configure_auto_save(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    session_id: Annotated[str, Field(description="Session identifier for the CSV data")],
    config: Annotated[AutoSaveConfig, Field(description="Auto-save configuration settings")],
) -> AutoSaveConfigResult:
    """Configure auto-save settings for a session.

    Enables and configures automatic saving of session data based on specified triggers and
    strategies. Use to ensure important work is automatically preserved during long data processing
    sessions.
    """
    try:
        await ctx.info(f"Configuring auto-save for session {session_id}")

        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(session_id)

        if not session:
            raise SessionNotFoundError(session_id)

        # Convert Pydantic model to dict for session
        config_dict = config.model_dump()

        # Enable auto-save with the provided configuration
        result = await session.enable_auto_save(config_dict)

        if result["success"]:
            await ctx.info(f"Auto-save configured successfully: {config.mode} mode")

            return AutoSaveConfigResult(
                success=True,
                config=config,
                previous_config=None,  # Would need to track previous config
            )
        await ctx.error(f"Auto-save configuration failed: {result.get('error', 'Unknown error')}")
        msg = f"Auto-save configuration failed: {result.get('error', 'Unknown error')}"
        raise ToolError(msg)

    except SessionNotFoundError as e:
        await ctx.error(f"Session not found: {e.details.get('session_id', session_id)}")
        msg = f"Session not found: {session_id}"
        raise ToolError(msg) from e
    except Exception as e:
        logger.error("Auto-save configuration failed for session %s: %s", session_id, str(e))
        await ctx.error(f"Auto-save configuration failed: {e}")
        msg = f"Auto-save configuration failed: {e}"
        raise ToolError(msg) from e


async def disable_auto_save(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    session_id: Annotated[str, Field(description="Session identifier for the CSV data")],
    *,
    perform_final_save: Annotated[
        bool, Field(description="Whether to save before disabling")
    ] = True,
) -> AutoSaveDisableResult:
    """Disable auto-save for a session.

    Stops automatic saving functionality. Optionally performs a final save before disabling to
    ensure current work is preserved.
    """
    try:
        await ctx.info(f"Disabling auto-save for session {session_id}")

        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(session_id)

        if not session:
            raise SessionNotFoundError(session_id)

        # Check if auto-save was enabled
        status_before = session.get_auto_save_status()
        was_enabled = status_before.get("enabled", False)

        # Perform final save if requested and auto-save was enabled
        final_save_performed = False
        if perform_final_save and was_enabled:
            save_result = await session.manual_save()
            final_save_performed = save_result.get("success", False)

        # Disable auto-save
        result = await session.disable_auto_save()

        if result["success"]:
            await ctx.info("Auto-save disabled successfully")

            return AutoSaveDisableResult(
                success=True,
                was_enabled=was_enabled,
                final_save_performed=final_save_performed,
            )
        await ctx.error(f"Failed to disable auto-save: {result.get('error', 'Unknown error')}")
        msg = f"Failed to disable auto-save: {result.get('error', 'Unknown error')}"
        raise ToolError(msg)

    except SessionNotFoundError as e:
        await ctx.error(f"Session not found: {e.details.get('session_id', session_id)}")
        msg = f"Session not found: {session_id}"
        raise ToolError(msg) from e
    except Exception as e:
        logger.error("Auto-save disable failed for session %s: %s", session_id, str(e))
        await ctx.error(f"Auto-save disable failed: {e}")
        msg = f"Auto-save disable failed: {e}"
        raise ToolError(msg) from e


async def get_auto_save_status(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    session_id: Annotated[str, Field(description="Session identifier for the CSV data")],
) -> AutoSaveStatusResult:
    """Get current auto-save status and configuration for a session.

    Returns detailed information about auto-save settings, last save time, and scheduling. Use to
    check auto-save configuration before relying on automatic data preservation.
    """
    try:
        await ctx.info(f"Getting auto-save status for session {session_id}")

        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(session_id)

        if not session:
            raise SessionNotFoundError(session_id)

        # Get auto-save status from session
        status_dict = session.get_auto_save_status()

        # Convert to structured response
        auto_save_status = AutoSaveStatus(
            enabled=status_dict.get("enabled", False),
            config=None,  # Would need to convert config format
            last_save_time=safe_str(value=status_dict.get("last_save_time"), default=""),
            last_save_success=safe_bool(value=status_dict.get("last_save_success"), default=False),
            next_save_time=safe_str(value=status_dict.get("next_save_time"), default=""),
        )

        await ctx.info("Auto-save status retrieved successfully")

        return AutoSaveStatusResult(
            success=True,
            status=auto_save_status,
        )

    except SessionNotFoundError as e:
        await ctx.error(f"Session not found: {e.details.get('session_id', session_id)}")
        msg = f"Session not found: {session_id}"
        raise ToolError(msg) from e
    except Exception as e:
        logger.error("Get auto-save status failed for session %s: %s", session_id, str(e))
        await ctx.error(f"Get auto-save status failed: {e}")
        msg = f"Get auto-save status failed: {e}"
        raise ToolError(msg) from e


async def trigger_manual_save(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    session_id: Annotated[str, Field(description="Session identifier for the CSV data")],
) -> ManualSaveResult:
    """Manually trigger a save operation for the session.

    Immediately saves the current session data using the configured auto-save settings. Use when you
    want to ensure data is saved at a specific point without waiting for automatic triggers.
    """
    try:
        await ctx.info(f"Triggering manual save for session {session_id}")

        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(session_id)

        if not session:
            raise SessionNotFoundError(session_id)

        # Trigger manual save
        result = await session.manual_save()

        if result["success"]:
            await ctx.info(f"Manual save completed: {result.get('file_path', 'Unknown path')}")

            return ManualSaveResult(
                success=True,
                file_path=safe_str(value=result.get("file_path"), default=""),
                rows_saved=safe_int(value=result.get("rows"), default=0),
                columns_saved=safe_int(value=result.get("columns"), default=0),
                save_time=safe_str(value=result.get("timestamp"), default=""),
            )
        await ctx.error(f"Manual save failed: {result.get('error', 'Unknown error')}")
        msg = f"Manual save failed: {result.get('error', 'Unknown error')}"
        raise ToolError(msg)

    except SessionNotFoundError as e:
        await ctx.error(f"Session not found: {e.details.get('session_id', session_id)}")
        msg = f"Session not found: {session_id}"
        raise ToolError(msg) from e
    except Exception as e:
        logger.error("Manual save failed for session %s: %s", session_id, str(e))
        await ctx.error(f"Manual save failed: {e}")
        msg = f"Manual save failed: {e}"
        raise ToolError(msg) from e


async def get_session_history(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    session_id: Annotated[str, Field(description="Session identifier for the CSV data")],
    limit: Annotated[
        int | None, Field(description="Maximum number of operations to return")
    ] = None,
) -> HistoryListResult:
    """Get operation history for a session.

    Returns chronological list of all operations performed on the session data. Use to review what
    changes have been made or to understand the data transformation workflow.
    """
    try:
        await ctx.info(f"Getting history for session {session_id}")

        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise SessionNotFoundError(session_id)

        # Get history from session
        history_result = session.get_history(limit)

        if history_result["success"]:
            operations = history_result.get("history", [])
            total = history_result.get("total", len(operations))

            await ctx.info(f"Retrieved {len(operations)} operations from session history")

            return HistoryListResult(
                success=True,
                operations=operations,
                total_operations=total,
                current_position=len(operations),
                can_undo=len(operations) > 0,
                can_redo=False,  # Would need to track redo stack
            )
        await ctx.error(f"Failed to get history: {history_result.get('error', 'Unknown error')}")
        msg = f"Failed to get history: {history_result.get('error', 'Unknown error')}"
        raise ToolError(msg)

    except SessionNotFoundError as e:
        await ctx.error(f"Session not found: {e.details.get('session_id', session_id)}")
        msg = f"Session not found: {session_id}"
        raise ToolError(msg) from e
    except Exception as e:
        logger.error("Get history failed for session %s: %s", session_id, str(e))
        await ctx.error(f"Get history failed: {e}")
        msg = f"Get history failed: {e}"
        raise ToolError(msg) from e


async def clear_session_history(
    ctx: Annotated[Context, Field(description="FastMCP context for session access")],
    session_id: Annotated[str, Field(description="Session identifier for the CSV data")],
    *,
    preserve_current_data: Annotated[
        bool, Field(description="Whether to keep current data state")
    ] = True,
) -> SessionOperationResult:
    """Clear the operation history for a session.

    Removes all operation history while optionally preserving the current data state. Use to start
    fresh or to reduce memory usage from extensive operation histories.
    """
    try:
        await ctx.info(f"Clearing history for session {session_id}")

        session_manager = get_session_manager()
        session = session_manager.get_or_create_session(session_id)

        if not session:
            raise SessionNotFoundError(session_id)

        operations_before = len(session.operations_history)

        # Clear history
        if session.history_manager:
            session.history_manager.clear_history()

        # Clear legacy history
        session.operations_history.clear()

        await ctx.info(f"Cleared {operations_before} operations from session history")

        return SessionOperationResult(
            success=True,
            message=f"Cleared {operations_before} operations from history",
            session_id=session_id,
            operation_type="clear_history",
            metadata={
                "operations_cleared": operations_before,
                "preserve_current_data": preserve_current_data,
            },
        )

    except SessionNotFoundError as e:
        await ctx.error(f"Session not found: {e.details.get('session_id', session_id)}")
        msg = f"Session not found: {session_id}"
        raise ToolError(msg) from e
    except Exception as e:
        logger.error("Clear history failed for session %s: %s", session_id, str(e))
        await ctx.error(f"Clear history failed: {e}")
        msg = f"Clear history failed: {e}"
        raise ToolError(msg) from e


# ============================================================================
# FASTMCP SERVER SETUP
# ============================================================================

# Create unified Session Management server
session_management_server = FastMCP(
    "DataBeak-SessionManagement",
    instructions="Unified session management server for DataBeak handling both operation history and auto-save functionality with comprehensive session lifecycle management",
)

# Register session management functions as MCP tools
session_management_server.tool(name="undo_operation")(undo_operation)
session_management_server.tool(name="redo_operation")(redo_operation)
session_management_server.tool(name="configure_auto_save")(configure_auto_save)
session_management_server.tool(name="disable_auto_save")(disable_auto_save)
session_management_server.tool(name="get_auto_save_status")(get_auto_save_status)
session_management_server.tool(name="trigger_manual_save")(trigger_manual_save)
session_management_server.tool(name="get_session_history")(get_session_history)
session_management_server.tool(name="clear_session_history")(clear_session_history)
