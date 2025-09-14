"""Core data session functionality separated from CSVSession."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..exceptions import NoDataLoadedError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DataSession:
    """Manages core DataFrame operations and metadata."""

    def __init__(self, session_id: str):
        """Initialize data session."""
        self.session_id = session_id
        self.df: pd.DataFrame | None = None
        self.original_df: pd.DataFrame | None = None
        self.file_path: str | None = None
        self.metadata: dict[str, Any] = {}  # type: ignore[explicit-any]  # Any justified: flexible session metadata
        self.created_at = datetime.now(timezone.utc)
        self.last_accessed = datetime.now(timezone.utc)

    def load_data(self, df: pd.DataFrame, file_path: str | None = None) -> None:
        """Load data into the session."""
        self.df = df.copy()
        self.original_df = df.copy()
        self.file_path = file_path
        self.update_access_time()

        # Update metadata
        self.metadata.update(
            {
                "file_path": file_path,
                "shape": df.shape,
                "columns": [str(col) for col in df.columns.tolist()],  # Ensure columns are strings
                "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
                "loaded_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def update_access_time(self) -> None:
        """Update the last accessed time."""
        self.last_accessed = datetime.now(timezone.utc)

    def get_data_info(self) -> dict[str, Any]:  # type: ignore[explicit-any]  # Any justified: flexible data info
        """Get information about the loaded data."""
        if self.df is None:
            raise NoDataLoadedError(self.session_id)

        memory_usage = self.df.memory_usage(deep=True).sum() / (1024 * 1024)

        return {
            "session_id": self.session_id,
            "shape": self.df.shape,
            "columns": [str(col) for col in self.df.columns.tolist()],  # Ensure columns are strings
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "memory_usage_mb": round(memory_usage, 2),
            "file_path": self.file_path,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
        }

    def validate_data_loaded(self) -> None:
        """Ensure data is loaded, raise exception if not."""
        if self.df is None:
            raise NoDataLoadedError(self.session_id)

    def clear_data(self) -> None:
        """Clear loaded data to free memory."""
        self.df = None
        self.original_df = None
        self.metadata.clear()

    def has_data(self) -> bool:
        """Check if data is loaded."""
        return self.df is not None

    def get_basic_stats(self) -> dict[str, Any]:  # type: ignore[explicit-any]  # Any justified: flexible stats data
        """Get basic statistics about the data."""
        if self.df is None:
            raise NoDataLoadedError(self.session_id)
        assert self.df is not None  # Type guard: validated above

        return {
            "row_count": len(self.df),
            "column_count": len(self.df.columns),
            "null_counts": self.df.isnull().sum().to_dict(),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        }
