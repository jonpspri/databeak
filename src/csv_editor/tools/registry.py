"""Tool registration system for MCP tools."""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

# Type for tool functions
ToolFunction = Callable[..., Awaitable[dict[str, Any]]]


class ToolRegistry:
    """Registry for MCP tools to reduce code duplication."""

    def __init__(self) -> None:
        """Initialize the tool registry."""
        self.tools: dict[str, ToolFunction] = {}
        self.tool_categories: dict[str, list[str]] = {
            "io": [],
            "data": [],
            "analytics": [],
            "validation": [],
            "history": [],
            "system": [],
            "row": [],
        }

    def register_tool(self, category: str, name: str) -> Callable[[ToolFunction], ToolFunction]:
        """Decorator to register a tool function."""

        def decorator(func: ToolFunction) -> ToolFunction:
            # Add to registry
            self.tools[name] = func
            if category in self.tool_categories:
                self.tool_categories[category].append(name)

            logger.debug(f"Registered tool '{name}' in category '{category}'")
            return func

        return decorator

    def get_tools_by_category(self, category: str) -> list[str]:
        """Get all tool names in a category."""
        return self.tool_categories.get(category, [])

    def register_all_tools(self, mcp: Any) -> None:
        """Register all tools with the FastMCP server."""
        for name, func in self.tools.items():
            # Register with FastMCP using the @mcp.tool decorator
            mcp.tool(func)
            logger.info(f"Registered MCP tool: {name}")

    def get_tool_count(self) -> int:
        """Get total number of registered tools."""
        return len(self.tools)

    def get_category_summary(self) -> dict[str, int]:
        """Get summary of tools by category."""
        return {cat: len(tools) for cat, tools in self.tool_categories.items()}


# Global registry instance
_tool_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def tool(category: str, name: str | None = None) -> Callable[[ToolFunction], ToolFunction]:
    """Decorator to register a tool with the registry."""

    def decorator(func: ToolFunction) -> ToolFunction:
        tool_name = name or func.__name__
        registry = get_tool_registry()
        return registry.register_tool(category, tool_name)(func)

    return decorator


def with_session_validation(func: ToolFunction) -> ToolFunction:
    """Decorator to add common session validation to tool functions."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        # Extract session_id from args/kwargs
        session_id = None
        if args:
            session_id = args[0]  # Assume first arg is session_id
        elif "session_id" in kwargs:
            session_id = kwargs["session_id"]

        if not session_id:
            return {
                "success": False,
                "error": {
                    "type": "MissingParameterError",
                    "message": "session_id is required",
                },
            }

        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Tool {func.__name__} failed: {e}")
            return {
                "success": False,
                "error": {
                    "type": e.__class__.__name__,
                    "message": str(e),
                },
            }

    return wrapper


def with_error_handling(func: ToolFunction) -> ToolFunction:
    """Decorator to add standardized error handling to tool functions."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            # Check if it's one of our custom exceptions
            if hasattr(e, "to_dict"):
                return {"success": False, "error": e.to_dict()}
            else:
                return {
                    "success": False,
                    "error": {
                        "type": "UnexpectedError",
                        "message": str(e),
                    },
                }

    return wrapper


def register_tools_from_module(module: Any, category: str) -> None:
    """Register all tools from a module with the given category."""
    registry = get_tool_registry()

    # Find all async functions that start with certain prefixes
    tool_prefixes = [
        "load_",
        "export_",
        "filter_",
        "get_",
        "add_",
        "remove_",
        "update_",
        "validate_",
        "analyze_",
        "calculate_",
        "create_",
    ]

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            callable(attr)
            and not attr_name.startswith("_")
            and any(attr_name.startswith(prefix) for prefix in tool_prefixes)
        ):
            # Register the tool
            registry.register_tool(category, attr_name)(attr)
            logger.info(f"Auto-registered tool '{attr_name}' in category '{category}'")
