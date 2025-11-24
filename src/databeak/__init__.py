"""DataBeak - MCP server for comprehensive CSV operations."""

__author__ = "Jonathan Springer"

import logging

from ._version import __version__
from .core.settings import get_settings
from .server import main

logging.getLogger("databeak").setLevel(get_settings().log_level)
logging.getLogger("mcp").setLevel(get_settings().log_level)

__all__ = ["__version__", "main"]
