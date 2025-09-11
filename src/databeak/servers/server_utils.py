"""Shared utilities for DataBeak MCP servers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ..exceptions import NoDataLoadedError, SessionNotFoundError
from ..models import get_session_manager

if TYPE_CHECKING:
    from ..models.csv_session import CSVSession


def get_session_data(session_id: str) -> tuple[CSVSession, pd.DataFrame]:
    """Get session and DataFrame, raising appropriate exceptions if not found.

    Args:
        session_id: The session identifier

    Returns:
        A tuple of (session, dataframe) where dataframe is guaranteed to be non-None

    Raises:
        SessionNotFoundError: If the session doesn't exist
        NoDataLoadedError: If the session has no data loaded
    """
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if not session:
        raise SessionNotFoundError(session_id)
    if not session.has_data():
        raise NoDataLoadedError(session_id)

    df = session.df
    if df is None:  # Type guard since has_data() was checked
        raise NoDataLoadedError(session_id)

    # At this point, both session and df are guaranteed to be non-None
    # The returned df is a reference to session._data_session.df
    # Any modifications to df will persist in the session
    return session, df
