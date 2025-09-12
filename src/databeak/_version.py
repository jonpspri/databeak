"""Version information for DataBeak."""

from __future__ import annotations

import importlib.metadata

try:
    __version__ = importlib.metadata.version("databeak")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development mode
    __version__ = "1.0.5-dev"

# Export for easy import
VERSION = __version__
