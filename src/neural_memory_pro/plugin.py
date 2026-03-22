"""Neural Memory Pro plugin — registers all Pro features."""

from __future__ import annotations

from typing import Any, Callable

from neural_memory.plugins.base import ProPlugin


class NMProPlugin(ProPlugin):
    """Pro plugin providing advanced retrieval, compression, and consolidation."""

    @property
    def name(self) -> str:
        return "neural-memory-pro"

    @property
    def version(self) -> str:
        from neural_memory_pro import __version__

        return __version__

    def get_retrieval_strategies(self) -> dict[str, Callable[..., Any]]:
        from neural_memory_pro.retrieval.cone_queries import cone_recall

        return {
            "cone": cone_recall,
        }

    def get_compression_fn(self) -> Callable[..., Any] | None:
        from neural_memory_pro.hyperspace.directional_compress import directional_compress

        return directional_compress

    def get_consolidation_strategies(self) -> dict[str, Callable[..., Any]]:
        from neural_memory_pro.consolidation.smart_merge import smart_merge

        return {
            "smart_merge": smart_merge,
        }
