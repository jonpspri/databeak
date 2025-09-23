"""DataBeak - MCP server for comprehensive CSV operations."""

__author__ = ["Jonathan Springer", "Santosh Ray"]

from ._version import __version__
from .core.settings import _settings as settings
from .server import main, mcp

__all__ = ["__version__", "main", "mcp", "settings"]
