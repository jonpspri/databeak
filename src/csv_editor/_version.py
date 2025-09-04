"""Version information for CSV Editor."""

from __future__ import annotations

import importlib.metadata

try:
    __version__ = importlib.metadata.version("csv-editor")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development mode
    __version__ = "1.0.2-dev"

# Export for easy import
VERSION = __version__
