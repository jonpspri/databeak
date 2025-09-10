"""Services package for DataBeak with dependency injection support.

This package contains service classes that implement business logic
with proper dependency injection for improved testability and reduced coupling.
"""

from .statistics_service import StatisticsService

__all__ = ["StatisticsService"]