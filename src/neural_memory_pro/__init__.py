"""Neural Memory Pro — DEPRECATED.

Pro features are now bundled in the main `neural-memory` package (v4.27+).
This package is no longer needed.

Upgrade:
    pip uninstall neural-memory-pro
    pip install neural-memory
    nmem pro activate YOUR_LICENSE_KEY
"""

from __future__ import annotations

import warnings

__version__ = "0.3.0"

warnings.warn(
    "neural-memory-pro is deprecated. Pro features are now bundled in "
    "'neural-memory' (v4.27+). Uninstall this package: "
    "pip uninstall neural-memory-pro",
    DeprecationWarning,
    stacklevel=2,
)

# Keep backward compat — imports still work via the old plugin system
from neural_memory_pro.infinitydb.engine import InfinityDB
from neural_memory_pro.storage_adapter import InfinityDBStorage

__all__ = [
    "InfinityDB",
    "InfinityDBStorage",
    "__version__",
    "auto_register",
]


def auto_register() -> None:
    """Entry point called by neural-memory plugin discovery."""
    from neural_memory.plugins import register

    from neural_memory_pro.plugin import NMProPlugin

    register(NMProPlugin())
