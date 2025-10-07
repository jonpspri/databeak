"""Main FastMCP server for DataBeak."""

from __future__ import annotations

# All MCP tools have been migrated to specialized server modules
import logging
from argparse import ArgumentParser
from pathlib import Path

from fastmcp import FastMCP

from databeak._version import __version__

# This module will tweak the JSON schema validator to accept relaxed types
from databeak.core.json_schema_validate import initialize_relaxed_validation

# Local imports
from databeak.servers.column_server import column_server
from databeak.servers.column_text_server import column_text_server
from databeak.servers.discovery_server import discovery_server
from databeak.servers.io_server import io_server
from databeak.servers.row_operations_server import row_operations_server
from databeak.servers.statistics_server import statistics_server
from databeak.servers.system_server import system_server
from databeak.servers.transformation_server import transformation_server
from databeak.servers.validation_server import validation_server
from databeak.utils.logging_config import set_correlation_id, setup_structured_logging

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
    except (PermissionError, OSError, UnicodeDecodeError):
        logger.exception("Error loading instructions")
        return "DataBeak MCP Server - Error loading instructions"


# Initialize relaxed JSON schema validation before creating server
initialize_relaxed_validation()

# Initialize FastMCP server
mcp = FastMCP("DataBeak", instructions=_load_instructions(), version=__version__)

# All tools have been migrated to specialized servers
# No direct tool registration needed - using server composition pattern

# Mount specialized servers
mcp.mount(system_server)
mcp.mount(io_server)
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
    """Start the DataBeak server."""
    parser = ArgumentParser(description="DataBeak")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="Transport method",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP/SSE transport")  # nosec B104
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

    run_args: dict[str, str | int] = {"transport": args.transport}
    if args.transport != "stdio":
        run_args["host"] = args.host
        run_args["port"] = args.port

    # Run the server
    mcp.run(**run_args)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
