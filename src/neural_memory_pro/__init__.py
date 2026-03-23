"""Neural Memory Pro — Advanced features for Neural Memory.

Auto-registers with neural-memory's plugin system on import.

Key exports:
- InfinityDB: Custom spatial database engine
- InfinityDBStorage: NeuralStorage adapter for InfinityDB
- cone_recall, smart_merge, directional_compress: Pro strategies
"""

from __future__ import annotations

__version__ = "0.2.0"

# Public API — lazy imports to avoid circular deps at import time
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
