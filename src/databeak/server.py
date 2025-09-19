"""Main FastMCP server for DataBeak."""

from __future__ import annotations

# All MCP tools have been migrated to specialized server modules
import logging
from pathlib import Path

from fastmcp import FastMCP

# Local imports
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
from .utils.logging_config import set_correlation_id, setup_structured_logging

# Configure structured logging
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _load_instructions() -> str:
    """Load instructions from the markdown file."""
    instructions_path = Path(__file__).parent / "instructions.md"
    try:
        return instructions_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Instructions file not found at %s", instructions_path)
        return "DataBeak MCP Server - Instructions file not available"
    except (PermissionError, OSError, UnicodeDecodeError) as e:
        logger.error("Error loading instructions: %s", e)
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
        "Starting DataBeak with %s transport",
        args.transport,
        extra={
            "transport": args.transport,
            "host": args.host if args.transport != "stdio" else None,
            "port": args.port if args.transport != "stdio" else None,
            "log_level": args.log_level,
            "server_id": server_correlation_id,
        },
    )

    # Run the server
    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
