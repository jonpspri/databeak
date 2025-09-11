"""Services package for DataBeak with dependency injection support.

This package contains service classes that implement business logic with proper dependency injection
for improved testability and reduced coupling. Also contains backend operation implementations 
used by server modules.
"""

from .statistics_service import StatisticsService

__all__ = ["StatisticsService", "data_operations", "transformation_operations"]
