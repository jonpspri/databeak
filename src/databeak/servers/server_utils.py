"""Shared utilities for DataBeak MCP servers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ..exceptions import NoDataLoadedError, SessionNotFoundError
from ..models import get_session_manager
from ..models.csv_session import CSVSession

if TYPE_CHECKING:
    pass


def get_session_data(session_id: str) -> CSVSession:
    """Get session for the given session_id.

    Args:
        session_id: The session identifier

    Returns:
        CSVSession

    Raises:
        SessionNotFoundError: If the session doesn't exist
    """
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if not session:
        raise SessionNotFoundError(session_id)

    return session
