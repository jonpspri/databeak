"""FastMCP data manipulation tool definitions for DataBeak.

NOTE: Most data tools have been migrated to specialized servers:
- transformation_server.py: filter_rows, sort_data, remove_duplicates, fill_missing_values
- column_server.py: select_columns, rename_columns, add_column, remove_columns, change_column_type, update_column
- column_text_server.py: replace_in_column, extract_from_column, split_column, transform_column_case, strip_column, fill_column_nulls

This module is retained for backwards compatibility and will be deprecated in a future version.
"""

from __future__ import annotations

from fastmcp import FastMCP


def register_data_tools(mcp: FastMCP) -> None:
    """Register data manipulation tools with FastMCP server.

    This function is now a no-op as all tools have been migrated to specialized servers. It is
    retained for backwards compatibility.
    """
    # All tools have been migrated to specialized servers
    pass
