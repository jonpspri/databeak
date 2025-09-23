"""Session utilities for defensive programming patterns.

This module provides standardized helper functions for safe session and DataFrame access, replacing
direct session.df patterns with proper error handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..exceptions import NoDataLoadedError
from ..core.session import _session_manager

if TYPE_CHECKING:
    import pandas as pd

    from ..core.session import DatabeakSession


def get_session_data(session_id: str) -> tuple[DatabeakSession, pd.DataFrame]:
    """Get session and DataFrame with comprehensive validation.

    This function replaces direct session.df access patterns with proper
    defensive programming, ensuring robust error handling and type safety.

    Returns:
        Tuple of (session, dataframe) - both guaranteed to be valid

    Raises:
        SessionNotFoundError: If session doesn't exist
        NoDataLoadedError: If session has no data loaded

    Example:
        # Instead of:
        session = databeak.session_manager.get_or_create_session(session_id)
        if not session.has_data():
            raise ToolError("No data")
        df = session.df
        assert df is not None

        # Use:
        session, df = get_session_data(session_id)

    """
    manager = _session_manager
    session = manager.get_or_create_session(session_id)

    # get_or_create_session always returns a session, so no need to check if not session
    if not session.has_data():
        raise NoDataLoadedError(session_id)

    df = session.df
    if df is None:  # Additional type guard for MyPy
        raise NoDataLoadedError(session_id)

    return session, df


def get_session_only(session_id: str) -> DatabeakSession:
    """Get session with validation but without requiring data.

    Use this when you need the session but data loading is optional.

    Returns:
        Valid DatabeakSession instance

    Raises:
        SessionNotFoundError: If session doesn't exist

    Example:
        # For operations that may create data or work without data
        session = get_session_only(session_id)
        if session.has_data():
            # Work with existing data
        else:
            # Initialize new data

    """
    manager = _session_manager
    # get_or_create_session always returns a session, so no need to check if not session
    return manager.get_or_create_session(session_id)


def validate_session_has_data(session: DatabeakSession, session_id: str) -> pd.DataFrame:
    """Validate that session has data and return DataFrame.

    Use this when you already have a session object and need to ensure data exists.

    Returns:
        Valid DataFrame instance

    Raises:
        NoDataLoadedError: If session has no data loaded

    Example:
        session = get_session_only(session_id)
        # ... some logic ...
        df = validate_session_has_data(session, session_id)

    """
    if not session.has_data():
        raise NoDataLoadedError(session_id)

    df = session.df
    if df is None:  # Additional type guard for MyPy
        raise NoDataLoadedError(session_id)

    return df


