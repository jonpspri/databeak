"""CSV resource definitions for MCP endpoints."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastmcp.exceptions import ToolError

# Local imports
from ..exceptions import NoDataLoadedError, SessionNotFoundError
from ..models import get_session_manager
from ..models.csv_session import CSVSession
from ..models.tool_responses import CellValueResult, RowDataResult
from ..models.typed_dicts import CsvDataResource
from ..services.data_operations import create_data_preview_with_indices
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _get_session_data(session_id: str) -> tuple[CSVSession, pd.DataFrame]:
    """Get validated session and DataFrame."""
    manager = get_session_manager()
    session = manager.get_or_create_session(session_id)

    if not session:
        raise SessionNotFoundError(session_id)
    if not session.has_data():
        raise NoDataLoadedError(session_id)

    df = session.df
    if df is None:  # Additional defensive check
        raise NoDataLoadedError(session_id)
    return session, df


async def get_cell_value(
    session_id: str,
    row_index: int,
    column: str | int,
) -> CellValueResult:
    """Get the value of a specific cell.

    Args:
        session_id: Session identifier
        row_index: Row index (0-based)
        column: Column name (str) or column index (int, 0-based)

    Returns:
        Dict with success status and cell value

    Example:
        get_cell_value("session123", 0, "name") -> {"success": True, "value": "John", "coordinates": {"row": 0, "column": "name"}}
        get_cell_value("session123", 2, 1) -> {"success": True, "value": 25, "coordinates": {"row": 2, "column": "age"}}
    """
    try:
        session, df = _get_session_data(session_id)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            raise ToolError(f"Row index {row_index} out of range (0-{len(df) - 1})")

        # Handle column specification
        if isinstance(column, int):
            # Column index
            if column < 0 or column >= len(df.columns):
                raise ToolError(f"Column index {column} out of range (0-{len(df.columns) - 1})")
            column_name = df.columns[column]
        else:
            # Column name
            if column not in df.columns:
                raise ToolError(f"Column '{column}' not found")
            column_name = column

        # Get the cell value
        cell_value = df.iloc[row_index][column_name]

        # Handle pandas/numpy types for JSON serialization
        if pd.isna(cell_value):
            cell_value = None
        elif hasattr(cell_value, "item"):
            cell_value = cell_value.item()

        return CellValueResult(
            value=cell_value,
            coordinates={"row": row_index, "column": column_name},
            data_type=str(df.dtypes[column_name]),
        )

    except Exception as e:
        logger.error(f"Error getting cell value: {e!s}")
        raise ToolError(f"Error: {e}") from e


async def get_row_data(
    session_id: str,
    row_index: int,
    columns: list[str] | None = None,
) -> RowDataResult:
    """Get data from a specific row.

    Args:
        session_id: Session identifier
        row_index: Row index (0-based)
        columns: Optional list of column names to include (None for all columns)

    Returns:
        Dict with success status and row data

    Example:
        get_row_data("session123", 0) -> {"success": True, "data": {"name": "John", "age": 30}, "row_index": 0}
        get_row_data("session123", 1, ["name", "age"]) -> {"success": True, "data": {"name": "Jane", "age": 25}}
    """
    try:
        session, df = _get_session_data(session_id)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            raise ToolError(f"Row index {row_index} out of range (0-{len(df) - 1})")

        # Get row data
        if columns is None:
            row_data = df.iloc[row_index].to_dict()
        else:
            # Validate columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ToolError(f"Columns not found: {missing_cols}")

            row_data = df.iloc[row_index][columns].to_dict()

        # Handle pandas/numpy types for JSON serialization
        for key, value in row_data.items():
            if pd.isna(value):
                row_data[key] = None
            elif hasattr(value, "item"):
                row_data[key] = value.item()

        return RowDataResult(
            session_id=session_id,
            row_index=row_index,
            data=row_data,
            columns=list(row_data.keys()),
        )

    except Exception as e:
        logger.error(f"Error getting row data: {e!s}")
        raise ToolError(f"Error: {e}") from e


# ============================================================================
# RESOURCE FUNCTIONS
# ============================================================================


async def get_csv_data(session_id: str) -> CsvDataResource | dict[str, str]:
    """Get current CSV data from a session with enhanced indexing."""
    session_manager = get_session_manager()
    session = session_manager.get_or_create_session(session_id)

    if not session or not session.has_data():
        return {"error": "Session not found or no data loaded"}

    # Use enhanced preview for better AI accessibility
    df = session.df
    if df is None:  # Additional defensive check
        return {"error": "No data available in session"}

    preview_data = create_data_preview_with_indices(df, 10)

    return CsvDataResource(
        session_id=session_id,
        shape=df.shape,
        preview=preview_data,
        columns_info={
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        },
    )


async def get_csv_schema(
    session_id: str,
) -> dict[str, str | list[str] | tuple[int, int] | dict[str, str]]:
    """Get CSV schema information."""
    session_manager = get_session_manager()
    session = session_manager.get_or_create_session(session_id)

    if not session or not session.has_data():
        return {"error": "Session not found or no data loaded"}

    df = session.df
    if df is None:  # Additional defensive check
        return {"error": "No data available in session"}

    return {
        "session_id": session_id,
        "columns": df.columns.tolist(),
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
    }


async def list_active_sessions() -> list[dict[str, str | int | bool | None]]:
    """List all active CSV sessions."""
    session_manager = get_session_manager()
    sessions = session_manager.list_sessions()
    return [s.dict() for s in sessions]


async def get_csv_cell(
    session_id: str, row_index: str, column: str
) -> dict[str, str | int | float | bool | None]:
    """Get data for a specific cell with coordinate information."""
    try:
        row_idx = int(row_index)
        # Try to convert column to int if it's numeric
        try:
            col_param: str | int = int(column)
        except ValueError:
            col_param = column

        result = await get_cell_value(session_id, row_idx, col_param)
        # Convert Pydantic model to dict for resource response
        return result.model_dump()
    except ValueError:
        return {"error": "Invalid row index - must be an integer"}
    except ToolError as e:
        return {"error": str(e)}


async def get_csv_row(
    session_id: str, row_index: str
) -> dict[str, str | int | float | bool | None]:
    """Get data for a specific row with all column values."""
    try:
        row_idx = int(row_index)
        result = await get_row_data(session_id, row_idx)
        # Convert Pydantic model to dict for resource response
        return result.model_dump()
    except ValueError:
        return {"error": "Invalid row index - must be an integer"}
    except ToolError as e:
        return {"error": str(e)}


async def get_csv_preview(session_id: str) -> dict[str, Any]:  # type: ignore[explicit-any]  # Any justified: MCP resource with complex unpacking
    """Get a preview of the CSV data with enhanced indexing and coordinate information."""
    session_manager = get_session_manager()
    session = session_manager.get_or_create_session(session_id)

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
