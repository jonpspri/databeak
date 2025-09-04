"""FastMCP row manipulation tool definitions for CSV Editor."""

from __future__ import annotations

from typing import Any

from fastmcp import Context  # noqa: TC002

from .transformations import (
    delete_row as _delete_row,
)
from .transformations import (
    get_cell_value as _get_cell_value,
)
from .transformations import (
    get_column_data as _get_column_data,
)
from .transformations import (
    get_row_data as _get_row_data,
)
from .transformations import (
    insert_row as _insert_row,
)
from .transformations import (
    set_cell_value as _set_cell_value,
)
from .transformations import (
    update_row as _update_row,
)


def register_row_tools(mcp: Any) -> None:
    """Register row manipulation tools with FastMCP server."""

    @mcp.tool
    async def get_cell_value(
        session_id: str, row_index: int, column: str | int, ctx: Context | None = None
    ) -> dict[str, Any]:
        """Get the value of a specific cell with precise coordinate targeting and comprehensive metadata.

        Essential tool for AI assistants to inspect individual cell values with full coordinate
        context. Part of the inspection workflow: get_data_summary → get_row_data → get_cell_value.

        Args:
            session_id: Session identifier for the active CSV data session
            row_index: Row index using 0-based indexing (0 = first row, N-1 = last row)
            column: Column targeting options:
                    - String: Column name (e.g., "name", "email", "status") - preferred for clarity
                    - Integer: Column index 0-based (0 = first column, N-1 = last column)

        Returns:
            Detailed cell information containing:
            - success: bool operation status
            - value: Actual cell value (None if null/NaN, preserves original type)
            - coordinates: {"row": index, "column": actual_column_name}
            - data_type: Column data type (int64, float64, object, datetime64, etc.)

        Examples:
            # Read by column name (recommended)
            get_cell_value("session123", 0, "name")    # → "John Doe"
            get_cell_value("session123", 5, "email")   # → "john@example.com" or None

            # Read by column index
            get_cell_value("session123", 2, 1)        # Third row, second column

            # Null value handling
            get_cell_value("session123", 1, "phone")  # → None if cell is empty/null

        AI Workflow Integration:
            1. **After get_data_summary()**: Use preview data to identify cells of interest
            2. **Before set_cell_value()**: Inspect current value to plan changes
            3. **With get_row_data()**: Compare individual cells to full row context
            4. **Error Debugging**: Use precise coordinates to troubleshoot issues

        Related Tools:
            ← get_data_summary(): Start with overview and preview data
            ← get_row_data(): Get full row context around this cell
            → set_cell_value(): Update this cell value
            → inspect_data_around(): Get surrounding data context
        """
        return await _get_cell_value(session_id, row_index, column, ctx)

    @mcp.tool
    async def set_cell_value(
        session_id: str,
        row_index: int,
        column: str | int,
        value: str | int | float | bool | None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Set the value of a specific cell with precise coordinate targeting and null value support.

        This function provides pixel-perfect cell editing capabilities optimized for AI assistants
        working with tabular data. Supports both column names and indices for flexible targeting.

        Args:
            session_id: Session identifier for the active CSV data session
            row_index: Row index using 0-based indexing (0 = first row, N-1 = last row)
            column: Column targeting options:
                    - String: Column name (e.g., "name", "email", "status")
                    - Integer: Column index 0-based (0 = first column, N-1 = last column)
            value: New cell value with full type support:
                   - Strings: "text", "email@example.com", ""
                   - Numbers: 42, 3.14, -100
                   - Booleans: true, false
                   - Null: null/None for missing data
                   - Automatically preserves type based on column context

        Returns:
            Detailed operation result containing:
            - success: bool operation status
            - coordinates: {"row": index, "column": actual_column_name}
            - old_value: Previous cell value (None if was null)
            - new_value: New cell value (None if setting to null)
            - data_type: Column data type information

        Examples:
            # Set by column name
            set_cell_value("session123", 0, "name", "Jane Smith")

            # Set by column index
            set_cell_value("session123", 2, 1, 25)

            # Set to null value
            set_cell_value("session123", 1, "email", null)
            set_cell_value("session123", 0, "phone", None)

            # Update status fields
            set_cell_value("session123", 3, "status", "completed")

        Error Conditions:
            - Invalid session_id or no data loaded
            - Row index out of range (0 to N-1)
            - Column name not found or column index out of range
            - Data type conversion errors (handled gracefully by pandas)

        AI Usage Tips:
            - Use column names for clarity when possible
            - Check coordinates in return value to verify targeting
            - Use get_cell_value first to inspect current value
            - Combine with get_row_data for context around changes
        """
        return await _set_cell_value(session_id, row_index, column, value, ctx)

    @mcp.tool
    async def get_row_data(
        session_id: str,
        row_index: int,
        columns: list[str] | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Get data from a specific row, optionally filtered by columns.

        Args:
            session_id: Session identifier
            row_index: Row index (0-based)
            columns: Optional list of column names to include (None for all columns)

        Returns:
            Dict with row data and metadata

        Examples:
            get_row_data("session123", 0) -> Get all data from first row
            get_row_data("session123", 1, ["name", "age"]) -> Get specific columns from second row
        """
        return await _get_row_data(session_id, row_index, columns, ctx)

    @mcp.tool
    async def get_column_data(
        session_id: str,
        column: str,
        start_row: int | None = None,
        end_row: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Get data from a specific column, optionally sliced by row range.

        Args:
            session_id: Session identifier
            column: Column name
            start_row: Starting row index (0-based, inclusive). None for beginning
            end_row: Ending row index (0-based, exclusive). None for end

        Returns:
            Dict with column data and metadata

        Examples:
            get_column_data("session123", "age") -> Get all values from "age" column
            get_column_data("session123", "name", 0, 5) -> Get first 5 values from "name" column
        """
        return await _get_column_data(session_id, column, start_row, end_row, ctx)

    @mcp.tool
    async def insert_row(
        session_id: str,
        row_index: int,
        data: (
            dict[str, str | int | float | bool | None] | list[str | int | float | bool | None] | str
        ),  # Accept string for Claude Code compatibility
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Insert a new row at the specified index with comprehensive null value and JSON string support.

        This function is optimized for AI assistants and supports multiple data input formats including
        automatic JSON string parsing for Claude Code compatibility.

        Args:
            session_id: Session identifier for the active CSV data session
            row_index: Index where to insert the row (0-based indexing). Use -1 to append at end
            data: Row data in multiple formats:
                  - Dict: {"column_name": value, ...} with all or partial columns
                  - List: [value1, value2, ...] matching column order exactly
                  - JSON string: Automatically parsed from Claude Code serialization

                  All formats support null/None values:
                  - JSON null → Python None → pandas NaN
                  - Missing dict keys → filled with None
                  - Explicit None values → preserved as pandas NaN

        Returns:
            Comprehensive operation result containing:
            - success: bool indicating operation status
            - operation: "insert_row" for tracking
            - row_index: Actual insertion index used
            - rows_before/rows_after: Data size changes
            - data_inserted: The actual row data that was inserted
            - columns: Current column list
            - session_id: Session identifier for continued operations

        Examples:
            # Standard dictionary insertion
            insert_row("session123", 1, {"name": "Alice", "age": 28, "city": "Boston"})

            # List insertion (append)
            insert_row("session123", -1, ["David", 40, "Miami"])

            # Null value support
            insert_row("session123", 0, {"name": "John", "age": null, "city": "NYC"})
            insert_row("session123", 2, ["Jane", None, "LA", None])

            # Claude Code JSON string (automatically handled)
            insert_row("session123", 1, '{"name": "Alice", "email": null, "status": "active"}')

            # Partial data (missing columns filled with None)
            insert_row("session123", 3, {"name": "Bob", "city": "Seattle"})  # age, email → None

        Error Conditions:
            - Invalid session_id or no data loaded
            - Row index out of valid range (0 to N for insertion)
            - Invalid JSON string format
            - List length mismatch with column count

        Coordinate System:
            - Uses 0-based indexing: row 0 is first, row N-1 is last
            - Insertion index N appends to end (same as row_index=-1)
            - All operations include precise coordinate tracking for AI assistance
        """
        return await _insert_row(session_id, row_index, data, ctx)

    @mcp.tool
    async def delete_row(
        session_id: str, row_index: int, ctx: Context | None = None
    ) -> dict[str, Any]:
        """Delete a row at the specified index.

        Args:
            session_id: Session identifier
            row_index: Row index to delete (0-based)

        Returns:
            Dict with success status and deletion info

        Example:
            delete_row("session123", 1) -> Delete second row
        """
        return await _delete_row(session_id, row_index, ctx)

    @mcp.tool
    async def update_row(
        session_id: str,
        row_index: int,
        data: dict[str, str | int | float | bool | None] | str,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Update specific columns in a row with comprehensive null value and Claude Code JSON string support.

        Provides selective column updates within a single row, supporting partial updates and
        automatic JSON string parsing. Optimized for AI assistants with detailed change tracking.

        Args:
            session_id: Session identifier for the active CSV data session
            row_index: Row index using 0-based indexing (0 = first row, N-1 = last row)
            data: Row update data in multiple formats:
                  - Dict: {"column_name": new_value, ...} for partial column updates
                  - JSON string: Automatically parsed from Claude Code serialization

                  All formats support null/None values:
                  - JSON null → Python None → pandas NaN
                  - Explicit None values → preserved as pandas NaN

        Returns:
            Detailed update result containing:
            - success: bool operation status
            - operation: "update_row" for tracking
            - row_index: The row that was updated
            - columns_updated: List of column names that were changed
            - old_values: Previous values for changed columns (None if was null)
            - new_values: New values for changed columns (None if set to null)
            - changes_made: Number of columns updated

        Examples:
            # Standard dictionary update
            update_row("session123", 0, {"age": 31, "city": "Boston"})

            # Set values to null
            update_row("session123", 1, {"phone": null, "email": null})

            # Partial update (only specified columns changed)
            update_row("session123", 2, {"status": "completed"})

        Claude Code JSON String Compatibility:
            Claude Code serializes parameters as JSON strings. This tool automatically handles:

            What Claude Code sends:
            ```json
            {
              "session_id": "abc123",
              "data": "{\\"status\\": \\"active\\", \\"notes\\": null}"
            }
            ```

            Automatically parsed to:
            ```python
            {
              "session_id": "abc123",
              "data": {"status": "active", "notes": None}
            }
            ```

        AI Workflow Integration:
            1. **After inspection**: Use get_row_data() to see current row state
            2. **Selective updates**: Update only specific columns that need changes
            3. **Change tracking**: Use old_values/new_values to verify updates
            4. **Null handling**: Set fields to null when data is missing or invalid

        Related Tools:
            ← get_row_data(): Inspect row before updating
            ← get_cell_value(): Check individual cell values
            → insert_row(): Add new rows instead of updating
            → set_cell_value(): Update single cell instead of multiple columns
        """
        return await _update_row(session_id, row_index, data, ctx)
