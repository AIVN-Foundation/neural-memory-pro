"""Neural Memory Pro plugin — registers all Pro features."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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

    def get_tools(self) -> list[dict[str, Any]]:
        from neural_memory_pro.mcp_tools import PRO_TOOL_SCHEMAS

        return list(PRO_TOOL_SCHEMAS)

    def get_tool_handler(self, tool_name: str) -> Callable[..., Any] | None:
        from neural_memory_pro.mcp_tools import TOOL_HANDLERS

        return TOOL_HANDLERS.get(tool_name)
