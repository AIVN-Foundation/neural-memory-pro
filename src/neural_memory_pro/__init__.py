"""Neural Memory Pro — Advanced features for Neural Memory.

Auto-registers with neural-memory's plugin system on import.
"""

from __future__ import annotations

__version__ = "0.1.0"


def auto_register() -> None:
    """Entry point called by neural-memory plugin discovery."""
    from neural_memory.plugins import register

    from neural_memory_pro.plugin import NMProPlugin

    register(NMProPlugin())
