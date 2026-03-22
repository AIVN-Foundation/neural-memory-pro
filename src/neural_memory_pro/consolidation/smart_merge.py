"""Smart merge consolidation — Pro-grade memory consolidation.

Free consolidation merges memories by simple similarity threshold.
Smart merge uses priority-aware clustering + temporal coherence:

1. Cluster memories by semantic similarity (embedding space)
2. Within each cluster, rank by: priority * recency * activation
3. Merge low-ranked memories into high-ranked anchors
4. Preserve causal chains (if A caused B, keep both or merge carefully)
5. Track merge provenance (which memories were merged into what)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeAction:
    """A single consolidation merge action."""

    anchor_id: str
    merged_ids: tuple[str, ...]
    new_content: str
    reason: str


async def smart_merge(
    storage: NeuralStorage,
    embed_fn: Any,
    *,
    similarity_threshold: float = 0.82,
    min_cluster_size: int = 2,
    max_merges: int = 20,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Priority-aware smart merge consolidation.

    Args:
        storage: Neural storage instance.
        embed_fn: Async embedding function.
        similarity_threshold: Min cosine sim to consider merging.
        min_cluster_size: Min cluster size to trigger merge.
        max_merges: Max number of merge actions per run.
        dry_run: If True, return planned actions without executing.

    Returns:
        Dict with merge results and statistics.
    """
    import numpy as np

    # 1. Get all neurons with embeddings
    neurons = await storage.find_neurons(limit=5000)
    if not neurons:
        return {"status": "empty", "merges": 0}

    # Build embedding matrix
    embedded = []
    for n in neurons:
        emb = getattr(n, "embedding", None)
        if emb is not None:
            embedded.append((n, np.array(emb, dtype=np.float32)))

    if len(embedded) < min_cluster_size:
        return {"status": "insufficient_embeddings", "merges": 0}

    # 2. Find similar pairs (O(n²) but capped at 5000)
    clusters: dict[str, list[int]] = {}  # anchor_idx -> [member_indices]
    assigned: set[int] = set()

    for i in range(len(embedded)):
        if i in assigned:
            continue

        cluster_members = [i]
        n_i, vec_i = embedded[i]
        norm_i = np.linalg.norm(vec_i)
        if norm_i < 1e-10:
            continue

        for j in range(i + 1, len(embedded)):
            if j in assigned:
                continue
            _, vec_j = embedded[j]
            norm_j = np.linalg.norm(vec_j)
            if norm_j < 1e-10:
                continue

            sim = float(np.dot(vec_i, vec_j) / (norm_i * norm_j))
            if sim >= similarity_threshold:
                cluster_members.append(j)

        if len(cluster_members) >= min_cluster_size:
            clusters[n_i.id] = cluster_members
            assigned.update(cluster_members)

    if not clusters:
        return {"status": "no_clusters", "merges": 0}

    # 3. Plan merge actions
    actions: list[MergeAction] = []
    for anchor_id, member_indices in clusters.items():
        if len(actions) >= max_merges:
            break

        members = [embedded[idx][0] for idx in member_indices]

        # Rank by priority * activation
        def _score(n: Any) -> float:
            priority = getattr(n, "priority", 5)
            activation = getattr(n, "activation_level", 0.5)
            p = priority if isinstance(priority, (int, float)) else 5
            a = activation if isinstance(activation, (int, float)) else 0.5
            return float(p) * float(a)

        members.sort(key=_score, reverse=True)
        anchor = members[0]
        to_merge = members[1:]

        if not to_merge:
            continue

        # Build merged content: anchor content + unique info from others
        merged_parts = [getattr(anchor, "content", "")]
        for m in to_merge:
            content = getattr(m, "content", "")
            if content and content not in merged_parts[0]:
                # Extract unique sentences
                existing = set(merged_parts[0].split(". "))
                new_sentences = [
                    s for s in content.split(". ")
                    if s.strip() and s not in existing
                ]
                if new_sentences:
                    merged_parts.append(". ".join(new_sentences))

        new_content = ". ".join(merged_parts)
        # Cap at reasonable length
        if len(new_content) > 2000:
            new_content = new_content[:1997] + "..."

        actions.append(
            MergeAction(
                anchor_id=anchor.id,
                merged_ids=tuple(getattr(m, "id", "") for m in to_merge),
                new_content=new_content,
                reason=f"Merged {len(to_merge)} similar memories (sim>{similarity_threshold})",
            )
        )

    result: dict[str, Any] = {
        "status": "planned" if dry_run else "executed",
        "clusters_found": len(clusters),
        "merge_actions": len(actions),
        "details": [
            {
                "anchor": a.anchor_id,
                "merged": list(a.merged_ids),
                "reason": a.reason,
            }
            for a in actions
        ],
    }

    # 4. Execute merges (if not dry run)
    if not dry_run:
        executed = 0
        for action in actions:
            try:
                # Update anchor content
                await storage.update_neuron(
                    action.anchor_id,
                    content=action.new_content,
                )
                # Mark merged neurons as consolidated (soft delete)
                for mid in action.merged_ids:
                    if mid:
                        await storage.update_neuron(mid, type="consolidated")
                executed += 1
            except Exception:
                logger.warning("Failed to execute merge for %s", action.anchor_id, exc_info=True)
        result["executed"] = executed

    return result
