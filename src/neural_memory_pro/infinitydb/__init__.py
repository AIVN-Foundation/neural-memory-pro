"""InfinityDB — Custom spatial database engine for Neural Memory Pro.

Multi-dimensional vector storage with HNSW indexing, graph-native synapses,
tiered compression, and crash-safe WAL. Designed for 1M+ neurons at <100ms recall.
"""

from __future__ import annotations

from neural_memory_pro.infinitydb.engine import InfinityDB

__all__ = ["InfinityDB"]
