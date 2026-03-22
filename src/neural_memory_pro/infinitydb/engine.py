"""InfinityDB — Main database engine.

Orchestrates vector store, HNSW index, and metadata store
to provide a unified neuron storage interface.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_memory_pro.infinitydb.file_format import BrainPaths, InfinityHeader
from neural_memory_pro.infinitydb.hnsw_index import HNSWIndex
from neural_memory_pro.infinitydb.metadata_store import MetadataStore
from neural_memory_pro.infinitydb.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    """UTC now as ISO string (naive, no tzinfo for SQLite compat)."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


class InfinityDB:
    """Custom spatial database engine for Neural Memory Pro.

    Provides neuron CRUD, vector similarity search, and metadata management.
    Uses memory-mapped vectors (numpy), HNSW index (hnswlib), and msgpack metadata.
    """

    def __init__(
        self,
        base_dir: str | Path,
        brain_id: str = "default",
        dimensions: int = 384,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._brain_id = brain_id
        self._dimensions = dimensions
        self._paths = BrainPaths(self._base_dir, brain_id)
        self._header = InfinityHeader(dimensions=dimensions)
        self._vectors = VectorStore(self._paths.vectors, dimensions)
        self._index = HNSWIndex(self._paths.index, dimensions)
        self._metadata = MetadataStore(self._paths.meta)
        self._is_open = False

    @property
    def brain_id(self) -> str:
        return self._brain_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def neuron_count(self) -> int:
        return self._metadata.count

    # --- Lifecycle ---

    async def open(self) -> None:
        """Open the database, loading all stores."""
        await asyncio.to_thread(self._open_sync)

    def _open_sync(self) -> None:
        self._paths.ensure_dirs()

        # Load or create header
        if self._paths.header.exists():
            data = self._paths.header.read_bytes()
            self._header = InfinityHeader.from_bytes(data)
            self._dimensions = self._header.dimensions
            # Recreate stores with correct dimensions
            self._vectors = VectorStore(self._paths.vectors, self._dimensions)
            self._index = HNSWIndex(self._paths.index, self._dimensions)
        else:
            self._header = InfinityHeader(dimensions=self._dimensions)
            self._paths.header.write_bytes(self._header.to_bytes())

        # Open stores
        self._metadata.open()
        self._vectors.open()

        # Open HNSW with capacity based on metadata count
        initial_cap = max(1024, self._metadata.count * 2)
        self._index.open(max_elements=initial_cap)

        self._is_open = True
        logger.info(
            "InfinityDB opened: brain=%s, neurons=%d, dims=%d",
            self._brain_id, self._metadata.count, self._dimensions,
        )

    async def close(self) -> None:
        """Close all stores and flush to disk."""
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        if not self._is_open:
            return
        self._flush_header()
        self._metadata.close()
        self._vectors.close()
        self._index.close()
        self._is_open = False
        logger.info("InfinityDB closed: brain=%s", self._brain_id)

    async def flush(self) -> None:
        """Flush all data to disk without closing."""
        await asyncio.to_thread(self._flush_sync)

    def _flush_sync(self) -> None:
        self._flush_header()
        self._metadata.flush()
        self._vectors.flush()
        self._index.save()

    def _flush_header(self) -> None:
        header = InfinityHeader(
            version=self._header.version,
            dimensions=self._dimensions,
            tier_config=self._header.tier_config,
            flags=self._header.flags,
            neuron_count=self._metadata.count,
            synapse_count=self._header.synapse_count,
        )
        self._paths.header.write_bytes(header.to_bytes())

    # --- Neuron CRUD ---

    async def add_neuron(
        self,
        content: str,
        *,
        neuron_id: str | None = None,
        neuron_type: str = "fact",
        embedding: list[float] | NDArray[np.float32] | None = None,
        priority: int = 5,
        activation_level: float = 1.0,
        tags: list[str] | None = None,
        ephemeral: bool = False,
    ) -> str:
        """Add a neuron with optional embedding vector.

        Returns the neuron ID.
        """
        nid = neuron_id or str(uuid.uuid4())
        now = _utcnow()

        # Store vector if embedding provided
        vec_slot = -1
        if embedding is not None:
            vec = np.asarray(embedding, dtype=np.float32)
            if vec.shape == (self._dimensions,):
                vec_slot = await asyncio.to_thread(self._vectors.add, vec)
                try:
                    await asyncio.to_thread(self._index.add, vec_slot, vec)
                except Exception:
                    # Rollback vector on index failure
                    await asyncio.to_thread(self._vectors.delete, vec_slot)
                    raise

        # Build metadata
        meta: dict[str, Any] = {
            "id": nid,
            "type": neuron_type,
            "content": content,
            "priority": priority,
            "activation_level": activation_level,
            "created_at": now,
            "updated_at": now,
            "accessed_at": now,
            "access_count": 0,
            "ephemeral": ephemeral,
            "tags": list(tags) if tags else [],
            "vec_slot": vec_slot,
        }

        meta_slot = vec_slot if vec_slot >= 0 else self._metadata.next_free_slot()
        try:
            await asyncio.to_thread(self._metadata.add, meta_slot, meta)
        except ValueError:
            # Rollback vector + index on metadata failure (e.g. duplicate ID)
            if vec_slot >= 0:
                await asyncio.to_thread(self._vectors.delete, vec_slot)
                await asyncio.to_thread(self._index.delete, vec_slot)
            raise
        return nid

    async def add_neurons_batch(
        self,
        neurons: list[dict[str, Any]],
    ) -> list[str]:
        """Batch insert neurons. Much faster than individual add_neuron calls.

        Each dict should have: content, and optionally: neuron_id, neuron_type,
        embedding (as numpy array), priority, tags, ephemeral.
        Returns list of neuron IDs.
        """
        return await asyncio.to_thread(self._add_neurons_batch_sync, neurons)

    def _add_neurons_batch_sync(self, neurons: list[dict[str, Any]]) -> list[str]:
        """Synchronous batch insert — no asyncio overhead."""
        now = _utcnow()
        ids: list[str] = []
        vec_slots: list[int] = []
        vec_arrays: list[NDArray[np.float32]] = []

        for neuron in neurons:
            nid = neuron.get("neuron_id") or str(uuid.uuid4())
            embedding = neuron.get("embedding")
            vec_slot = -1

            if embedding is not None:
                vec = np.asarray(embedding, dtype=np.float32)
                if vec.shape == (self._dimensions,):
                    vec_slot = self._vectors.add(vec)
                    vec_slots.append(vec_slot)
                    vec_arrays.append(vec)

            meta: dict[str, Any] = {
                "id": nid,
                "type": neuron.get("neuron_type", "fact"),
                "content": neuron.get("content", ""),
                "priority": neuron.get("priority", 5),
                "activation_level": neuron.get("activation_level", 1.0),
                "created_at": now,
                "updated_at": now,
                "accessed_at": now,
                "access_count": 0,
                "ephemeral": neuron.get("ephemeral", False),
                "tags": list(neuron.get("tags", [])),
                "vec_slot": vec_slot,
            }
            slot = vec_slot if vec_slot >= 0 else self._metadata.next_free_slot()
            self._metadata.add(slot, meta)
            ids.append(nid)

        # Batch add to HNSW index AFTER metadata is committed
        if vec_slots and vec_arrays:
            vectors = np.stack(vec_arrays)
            self._index.add_batch(vec_slots, vectors)

        return ids

    async def get_neuron(self, neuron_id: str) -> dict[str, Any] | None:
        """Get a neuron by ID."""
        result = await asyncio.to_thread(self._metadata.get_by_id, neuron_id)
        if result is None:
            return None
        _, meta = result
        return dict(meta)

    async def find_neurons(
        self,
        *,
        neuron_type: str | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[str, str] | None = None,
        limit: int = 100,
        offset: int = 0,
        ephemeral: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Find neurons matching filters."""
        results = await asyncio.to_thread(
            self._metadata.find,
            neuron_type=neuron_type,
            content_contains=content_contains,
            content_exact=content_exact,
            time_range=time_range,
            limit=limit,
            offset=offset,
            ephemeral=ephemeral,
        )
        return [dict(meta) for _, meta in results]

    # Allowlist of fields that can be updated via update_neuron
    _UPDATABLE_FIELDS = frozenset({
        "content", "type", "priority", "activation_level", "tags",
        "ephemeral", "accessed_at", "access_count", "updated_at",
    })

    async def update_neuron(
        self,
        neuron_id: str,
        *,
        content: str | None = None,
        neuron_type: str | None = None,
        priority: int | None = None,
        activation_level: float | None = None,
        embedding: list[float] | NDArray[np.float32] | None = None,
        tags: list[str] | None = None,
        ephemeral: bool | None = None,
    ) -> bool:
        """Update a neuron's metadata and/or vector."""
        result = await asyncio.to_thread(self._metadata.get_by_id, neuron_id)
        if result is None:
            return False

        slot, meta = result
        updates: dict[str, Any] = {"updated_at": _utcnow()}

        if content is not None:
            updates["content"] = content
        if neuron_type is not None:
            updates["type"] = neuron_type
        if priority is not None:
            updates["priority"] = priority
        if activation_level is not None:
            updates["activation_level"] = activation_level
        if tags is not None:
            updates["tags"] = list(tags)
        if ephemeral is not None:
            updates["ephemeral"] = ephemeral

        # Update vector if new embedding provided
        if embedding is not None:
            vec = np.asarray(embedding, dtype=np.float32)
            if vec.shape == (self._dimensions,):
                old_slot = meta.get("vec_slot", -1)
                if old_slot >= 0:
                    await asyncio.to_thread(self._index.delete, old_slot)
                    await asyncio.to_thread(self._vectors.update, old_slot, vec)
                    await asyncio.to_thread(self._index.add, old_slot, vec)
                else:
                    new_slot = await asyncio.to_thread(self._vectors.add, vec)
                    await asyncio.to_thread(self._index.add, new_slot, vec)
                    updates["vec_slot"] = new_slot

        await asyncio.to_thread(self._metadata.update, slot, updates)
        return True

    async def delete_neuron(self, neuron_id: str) -> bool:
        """Delete a neuron and its vector."""
        result = await asyncio.to_thread(self._metadata.get_by_id, neuron_id)
        if result is None:
            return False

        slot, meta = result
        vec_slot = meta.get("vec_slot", -1)

        if vec_slot >= 0:
            await asyncio.to_thread(self._vectors.delete, vec_slot)
            await asyncio.to_thread(self._index.delete, vec_slot)

        await asyncio.to_thread(self._metadata.delete, slot)
        return True

    # --- Vector Search ---

    async def search_similar(
        self,
        query_vector: list[float] | NDArray[np.float32],
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for k most similar neurons by vector.

        Returns list of dicts with neuron metadata + similarity score.
        """
        vec = np.asarray(query_vector, dtype=np.float32)
        if vec.shape != (self._dimensions,):
            return []

        slot_ids, distances = await asyncio.to_thread(self._index.search, vec, k)

        results = []
        for slot_id, dist in zip(slot_ids, distances):
            meta = await asyncio.to_thread(self._metadata.get_by_slot, slot_id)
            if meta is not None:
                results.append({
                    **meta,
                    "similarity": round(1.0 - dist, 4),  # cosine: dist = 1 - sim
                    "distance": round(dist, 6),
                })

        return results

    async def search_similar_batch(
        self,
        query_vectors: NDArray[np.float32],
        k: int = 10,
    ) -> list[list[dict[str, Any]]]:
        """Batch search for multiple query vectors."""
        all_labels, all_distances = await asyncio.to_thread(
            self._index.search_batch, query_vectors, k
        )

        batch_results = []
        for i in range(len(query_vectors)):
            results = []
            for j in range(all_labels.shape[1]):
                slot_id = int(all_labels[i][j])
                dist = float(all_distances[i][j])
                meta = await asyncio.to_thread(self._metadata.get_by_slot, slot_id)
                if meta is not None:
                    results.append({
                        **meta,
                        "similarity": round(1.0 - dist, 4),
                        "distance": round(dist, 6),
                    })
            batch_results.append(results)

        return batch_results

    # --- Stats ---

    async def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        return {
            "brain_id": self._brain_id,
            "neuron_count": self._metadata.count,
            "vector_count": self._vectors.count,
            "index_count": self._index.count,
            "dimensions": self._dimensions,
            "is_open": self._is_open,
        }

    # --- Suggest ---

    async def suggest_neurons(
        self,
        prefix: str,
        type_filter: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Suggest neurons by content prefix."""
        return await asyncio.to_thread(
            self._metadata.suggest, prefix, type_filter, limit
        )
