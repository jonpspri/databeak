"""Session lifecycle management separated from CSVSession."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from ..exceptions import SessionExpiredError

logger = logging.getLogger(__name__)


class SessionLifecycle:
    """Manages session TTL and expiration logic."""

    def __init__(self, session_id: str, ttl_minutes: int = 60):
        """Initialize session lifecycle manager."""
        self.session_id = session_id
        self.ttl = timedelta(minutes=ttl_minutes)
        self.created_at = datetime.now(timezone.utc)
        self.last_accessed = datetime.now(timezone.utc)

    def update_access_time(self) -> None:
        """Update the last accessed time."""
        self.last_accessed = datetime.now(timezone.utc)

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now(timezone.utc) - self.last_accessed > self.ttl

    def validate_not_expired(self) -> None:
        """Raise exception if session is expired."""
        if self.is_expired():
            raise SessionExpiredError(self.session_id)

    def get_remaining_ttl(self) -> timedelta:
        """Get remaining time until expiration."""
        elapsed = datetime.now(timezone.utc) - self.last_accessed
        return max(timedelta(0), self.ttl - elapsed)

    def extend_ttl(self, additional_minutes: int) -> None:
        """Extend the session TTL."""
        self.ttl += timedelta(minutes=additional_minutes)
        logger.info(f"Extended TTL for session {self.session_id} by {additional_minutes} minutes")

    def get_lifecycle_info(self) -> dict[str, Any]:
        """Get session lifecycle information."""
        remaining = self.get_remaining_ttl()

        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "ttl_minutes": int(self.ttl.total_seconds() / 60),
            "remaining_minutes": int(remaining.total_seconds() / 60),
            "is_expired": self.is_expired(),
        }
