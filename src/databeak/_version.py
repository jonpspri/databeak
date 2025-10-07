"""Version information for DataBeak."""

from __future__ import annotations

import importlib.metadata

__version__ = importlib.metadata.version("databeak")

# Export for easy import
VERSION = __version__
