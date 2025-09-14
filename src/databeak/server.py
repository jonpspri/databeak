"""Main FastMCP server for DataBeak."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Local imports
from .models import get_session_manager
from .servers.column_server import column_server
from .servers.column_text_server import column_text_server
from .servers.discovery_server import discovery_server
from .servers.history_server import history_server
from .servers.io_server import io_server
from .servers.row_operations_server import row_operations_server
from .servers.statistics_server import statistics_server
from .servers.system_server import system_server
from .servers.transformation_server import transformation_server
from .servers.validation_server import validation_server
from .services.data_operations import create_data_preview_with_indices

# All MCP tools have been migrated to specialized server modules
from .services.transformation_operations import get_cell_value as _get_cell_value
from .services.transformation_operations import get_row_data as _get_row_data
from .utils.logging_config import get_logger, set_correlation_id, setup_structured_logging

# Configure structured logging
logger = get_logger(__name__)


def _load_instructions() -> str:
    """Load instructions from the markdown file."""
    instructions_path = Path(__file__).parent / "instructions.md"
    try:
        return instructions_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(f"Instructions file not found at {instructions_path}")
        return "DataBeak MCP Server - Instructions file not available"
    except (PermissionError, OSError, UnicodeDecodeError) as e:
        logger.error(f"Error loading instructions: {e}")
        return "DataBeak MCP Server - Error loading instructions"


# Initialize FastMCP server
mcp = FastMCP("DataBeak", instructions=_load_instructions())

# All tools have been migrated to specialized servers
# No direct tool registration needed - using server composition pattern

# Mount specialized servers
mcp.mount(system_server)
mcp.mount(io_server)
mcp.mount(history_server)
mcp.mount(row_operations_server)
mcp.mount(statistics_server)
mcp.mount(discovery_server)
mcp.mount(validation_server)
mcp.mount(transformation_server)
mcp.mount(column_server)
mcp.mount(column_text_server)

# ============================================================================
# RESOURCES
# ============================================================================


@mcp.resource("csv://{session_id}/data")
async def get_csv_data(session_id: str) -> dict[str, Any]:  # type: ignore[misc]  # Any justified: MCP resource flexible response  # type: ignore[misc]  # Any justified: MCP resource flexible response
    """Get current CSV data from a session with enhanced indexing."""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if not session or not session.has_data():
        return {"error": "Session not found or no data loaded"}

    # Use enhanced preview for better AI accessibility
    df = session.df
    if df is None:  # Additional defensive check
        return {"error": "No data available in session"}

    preview_data = create_data_preview_with_indices(df, 10)

    return {
        "session_id": session_id,
        "shape": df.shape,
        "preview": preview_data,
        "columns_info": {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        },
    }


@mcp.resource("csv://{session_id}/schema")
async def get_csv_schema(session_id: str) -> dict[str, Any]:  # type: ignore[misc]  # Any justified: MCP resource flexible response
    """Get CSV schema information."""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if not session or not session.has_data():
        return {"error": "Session not found or no data loaded"}

    df = session.df
    if df is None:  # Additional defensive check
        return {"error": "No data available in session"}

    return {
        "session_id": session_id,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
    }


@mcp.resource("sessions://active")
async def list_active_sessions() -> list[dict[str, Any]]:  # type: ignore[misc]  # Any justified: MCP resource list response
    """List all active CSV sessions."""
    session_manager = get_session_manager()
    sessions = session_manager.list_sessions()
    return [s.dict() for s in sessions]


@mcp.resource("csv://{session_id}/cell/{row_index}/{column}")
async def get_csv_cell(session_id: str, row_index: str, column: str) -> dict[str, Any]:  # type: ignore[misc]  # Any justified: MCP resource flexible response
    """Get data for a specific cell with coordinate information."""
    try:
        row_idx = int(row_index)
        # Try to convert column to int if it's numeric
        try:
            col_param: str | int = int(column)
        except ValueError:
            col_param = column

        result = await _get_cell_value(session_id, row_idx, col_param)
        # Convert Pydantic model to dict for resource response
        return result.model_dump()
    except ValueError:
        return {"error": "Invalid row index - must be an integer"}
    except ToolError as e:
        return {"error": str(e)}


@mcp.resource("csv://{session_id}/row/{row_index}")
async def get_csv_row(session_id: str, row_index: str) -> dict[str, Any]:  # type: ignore[misc]  # Any justified: MCP resource flexible response
    """Get data for a specific row with all column values."""
    try:
        row_idx = int(row_index)
        result = await _get_row_data(session_id, row_idx)
        # Convert Pydantic model to dict for resource response
        return result.model_dump()
    except ValueError:
        return {"error": "Invalid row index - must be an integer"}
    except ToolError as e:
        return {"error": str(e)}


@mcp.resource("csv://{session_id}/preview")
async def get_csv_preview(session_id: str) -> dict[str, Any]:  # type: ignore[misc]  # Any justified: MCP resource flexible response
    """Get a preview of the CSV data with enhanced indexing and coordinate information."""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if not session or not session.has_data():
        return {"error": "Session not found or no data loaded"}

    df = session.df
    if df is None:  # Additional defensive check
        return {"error": "No data available in session"}

    preview_data = create_data_preview_with_indices(df, 10)

    return {
        "session_id": session_id,
        "coordinate_system": {
            "description": "Uses 0-based indexing for both rows and columns",
            "row_indexing": "0 to N-1 where N is total rows",
            "column_indexing": "Use column names (strings) or 0-based column indices (integers)",
        },
        **preview_data,
    }


# ============================================================================
# PROMPTS
# ============================================================================


@mcp.prompt
def analyze_csv_prompt(session_id: str, analysis_type: str = "summary") -> str:
    """Generate a prompt to analyze CSV data."""
    return f"""Please analyze the CSV data in session {session_id}.

Analysis type: {analysis_type}

Provide insights about:
1. Data quality and completeness
2. Statistical patterns
3. Potential issues or anomalies
4. Recommended transformations or cleanups
"""


@mcp.prompt
def data_cleaning_prompt(session_id: str) -> str:
    """Generate a prompt for data cleaning suggestions."""
    return f"""Review the data in session {session_id} and suggest cleaning operations.

Consider:
- Missing values and how to handle them
- Duplicate rows
- Data type conversions needed
- Outliers that may need attention
- Column naming conventions
"""


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main() -> None:
    """Main entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="DataBeak")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="Transport method",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP/SSE transport")  # nosec B104  # noqa: S104
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP/SSE transport")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup structured logging
    setup_structured_logging(args.log_level)

    # Set server-level correlation ID
    server_correlation_id = set_correlation_id()

    logger.info(
        f"Starting DataBeak with {args.transport} transport",
        transport=args.transport,
        host=args.host if args.transport != "stdio" else None,
        port=args.port if args.transport != "stdio" else None,
        log_level=args.log_level,
        server_id=server_correlation_id,
    )

    # Run the server
    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
