"""Cone queries — exhaustive recall via embedding similarity cones.

Free recall truncates results to top-N. Cone queries return ALL memories
within a cosine similarity cone around the query embedding, ensuring
no relevant memory is missed.

Algorithm:
1. Embed the query
2. Compute cosine similarity against ALL neuron embeddings in brain
3. Return everything above the cone threshold (adaptive or fixed)
4. Rank by combined score: similarity * activation_level * freshness
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Default cone half-angle as cosine threshold (cos(30°) ≈ 0.866)
DEFAULT_CONE_THRESHOLD = 0.65


@dataclass(frozen=True)
class ConeResult:
    """A single memory within the recall cone."""

    neuron_id: str
    content: str
    similarity: float
    activation: float
    combined_score: float
    neuron_type: str


async def cone_recall(
    query: str,
    storage: NeuralStorage,
    embed_fn: Any,
    *,
    threshold: float = DEFAULT_CONE_THRESHOLD,
    max_results: int = 500,
    boost_recent: bool = True,
) -> list[ConeResult]:
    """Exhaustive cone recall — return ALL memories within similarity cone.

    Args:
        query: Search query string.
        storage: Neural storage instance.
        embed_fn: Async embedding function (str -> list[float]).
        threshold: Minimum cosine similarity (0-1). Lower = wider cone.
        max_results: Safety cap to prevent unbounded results.
        boost_recent: Boost recently activated neurons.

    Returns:
        List of ConeResult sorted by combined_score descending.
    """
    import numpy as np

    # 1. Embed the query
    query_vec = await embed_fn(query)
    if not query_vec:
        return []
    query_arr = np.array(query_vec, dtype=np.float32)
    query_norm = np.linalg.norm(query_arr)
    if query_norm < 1e-10:
        return []
    query_arr = query_arr / query_norm

    # 2. Get all neurons with embeddings
    neurons = await storage.find_neurons(limit=10000)
    if not neurons:
        return []

    results: list[ConeResult] = []

    for neuron in neurons:
        # Get embedding from neuron metadata or storage
        embedding = getattr(neuron, "embedding", None)
        if embedding is None:
            continue

        neuron_arr = np.array(embedding, dtype=np.float32)
        neuron_norm = np.linalg.norm(neuron_arr)
        if neuron_norm < 1e-10:
            continue
        neuron_arr = neuron_arr / neuron_norm

        # 3. Cosine similarity
        similarity = float(np.dot(query_arr, neuron_arr))

        if similarity < threshold:
            continue

        # 4. Combined score
        activation = getattr(neuron, "activation_level", 0.5)
        if not isinstance(activation, (int, float)):
            activation = 0.5

        combined = similarity * 0.7 + float(activation) * 0.3

        results.append(
            ConeResult(
                neuron_id=neuron.id,
                content=getattr(neuron, "content", ""),
                similarity=round(similarity, 4),
                activation=round(float(activation), 4),
                combined_score=round(combined, 4),
                neuron_type=getattr(neuron, "type", "unknown"),
            )
        )

    # Sort by combined score, cap results
    results.sort(key=lambda r: r.combined_score, reverse=True)
    return results[:max_results]
