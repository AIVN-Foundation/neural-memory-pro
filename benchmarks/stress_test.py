"""InfinityDB Stress Test — 1M and 2M neuron benchmarks.

Run overnight to validate performance at scale:
    python benchmarks/stress_test.py

Results are written to benchmarks/results/<timestamp>.json

Metrics captured:
- Insert throughput (neurons/sec, batch and individual)
- Vector search latency (p50, p95, p99 at various k values)
- Graph traversal latency (BFS at various depths)
- Multi-dimensional query latency (RRF fusion)
- Disk size and compression ratio
- Memory usage (RSS)
- Flush/checkpoint durability
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neural_memory_pro.infinitydb.engine import InfinityDB  # noqa: E402
from neural_memory_pro.infinitydb.query_planner import QueryPlan  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("stress_test")

# ── Configuration ──

DIMENSIONS = 384  # production embedding size
BATCH_SIZE = 5000
SEARCH_K_VALUES = [10, 50, 100]
BFS_DEPTHS = [1, 2, 3]
SEARCH_ITERATIONS = 100  # number of search queries per k value
SYNAPSE_RATIO = 3  # average synapses per neuron
FIBER_COUNT = 50  # number of fibers to create

NEURON_TYPES = ["fact", "decision", "error", "insight", "preference",
                "workflow", "instruction", "concept", "entity", "pattern"]


def _get_rss_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback for Windows without psutil
        try:
            import ctypes
            import ctypes.wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(pmc), pmc.cb
            )
            return pmc.WorkingSetSize / 1024 / 1024
        except Exception:
            return 0.0


def _dir_size_mb(path: Path) -> float:
    """Get total directory size in MB."""
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total / 1024 / 1024


def _percentile(values: list[float], p: float) -> float:
    """Calculate percentile from a list of values."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * p / 100)
    idx = min(idx, len(sorted_v) - 1)
    return sorted_v[idx]


async def benchmark_insert(
    db: InfinityDB,
    neuron_count: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Benchmark batch neuron insertion."""
    logger.info("=== INSERT BENCHMARK: %d neurons ===", neuron_count)
    results: dict[str, Any] = {"neuron_count": neuron_count}

    t0 = time.perf_counter()
    total_inserted = 0
    batch_times: list[float] = []

    for batch_start in range(0, neuron_count, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, neuron_count)
        batch_count = batch_end - batch_start

        neurons: list[dict[str, Any]] = []
        for i in range(batch_start, batch_end):
            vec = rng.standard_normal(DIMENSIONS).astype(np.float32)
            vec /= np.linalg.norm(vec)  # normalize
            neurons.append({
                "neuron_id": f"n{i:08d}",
                "content": f"Benchmark neuron {i} with random content for stress testing. "
                           f"This simulates real-world memory content with varying lengths. "
                           f"Priority={i % 10 + 1}, type={NEURON_TYPES[i % len(NEURON_TYPES)]}.",
                "neuron_type": NEURON_TYPES[i % len(NEURON_TYPES)],
                "priority": i % 10 + 1,
                "embedding": vec,
                "tags": [f"tag-{i % 20}", f"group-{i % 100}"],
                "ephemeral": i % 50 == 0,
            })

        bt0 = time.perf_counter()
        await db.add_neurons_batch(neurons)
        bt1 = time.perf_counter()

        batch_times.append(bt1 - bt0)
        total_inserted += batch_count

        if total_inserted % 100_000 == 0 or total_inserted == neuron_count:
            elapsed = time.perf_counter() - t0
            rate = total_inserted / elapsed
            rss = _get_rss_mb()
            logger.info(
                "  Inserted %d/%d (%.0f neurons/sec, RSS=%.0fMB)",
                total_inserted, neuron_count, rate, rss,
            )

    total_time = time.perf_counter() - t0
    results["total_seconds"] = round(total_time, 2)
    results["neurons_per_second"] = round(neuron_count / total_time, 1)
    results["batch_p50_ms"] = round(_percentile(batch_times, 50) * 1000, 2)
    results["batch_p95_ms"] = round(_percentile(batch_times, 95) * 1000, 2)
    results["batch_p99_ms"] = round(_percentile(batch_times, 99) * 1000, 2)
    results["rss_mb_after_insert"] = round(_get_rss_mb(), 1)

    logger.info(
        "  INSERT DONE: %d neurons in %.1fs (%.0f/sec), p50=%.1fms, p95=%.1fms",
        neuron_count, total_time, results["neurons_per_second"],
        results["batch_p50_ms"], results["batch_p95_ms"],
    )
    return results


async def benchmark_synapses(
    db: InfinityDB,
    neuron_count: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Benchmark synapse insertion."""
    synapse_count = neuron_count * SYNAPSE_RATIO
    logger.info("=== SYNAPSE BENCHMARK: %d synapses ===", synapse_count)

    t0 = time.perf_counter()
    inserted = 0
    errors = 0

    for i in range(synapse_count):
        src_idx = rng.integers(0, neuron_count)
        tgt_idx = rng.integers(0, neuron_count)
        if src_idx == tgt_idx:
            tgt_idx = (tgt_idx + 1) % neuron_count

        try:
            await db.add_synapse(
                f"n{src_idx:08d}",
                f"n{tgt_idx:08d}",
                edge_type="related" if i % 3 != 0 else "causal",
                weight=rng.uniform(0.1, 1.0),
            )
            inserted += 1
        except Exception:
            errors += 1

        if inserted % 100_000 == 0 and inserted > 0:
            elapsed = time.perf_counter() - t0
            logger.info("  Synapses: %d/%d (%.0f/sec)", inserted, synapse_count, inserted / elapsed)

    total_time = time.perf_counter() - t0
    results = {
        "synapse_count": inserted,
        "errors": errors,
        "total_seconds": round(total_time, 2),
        "synapses_per_second": round(inserted / total_time, 1) if total_time > 0 else 0,
    }
    logger.info(
        "  SYNAPSE DONE: %d in %.1fs (%.0f/sec), %d errors",
        inserted, total_time, results["synapses_per_second"], errors,
    )
    return results


async def benchmark_fibers(
    db: InfinityDB,
    neuron_count: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Benchmark fiber creation and neuron association."""
    logger.info("=== FIBER BENCHMARK: %d fibers ===", FIBER_COUNT)

    t0 = time.perf_counter()
    for i in range(FIBER_COUNT):
        # Each fiber gets 100-1000 random neurons
        count = rng.integers(100, 1000)
        neuron_ids = [f"n{rng.integers(0, neuron_count):08d}" for _ in range(count)]
        await db.add_fiber(
            name=f"fiber-{i}",
            fiber_id=f"f{i:04d}",
            fiber_type="cluster" if i % 2 == 0 else "topic",
            description=f"Benchmark fiber {i}",
            neuron_ids=neuron_ids,
        )

    total_time = time.perf_counter() - t0
    results = {
        "fiber_count": FIBER_COUNT,
        "total_seconds": round(total_time, 2),
    }
    logger.info("  FIBER DONE: %d in %.1fs", FIBER_COUNT, total_time)
    return results


async def benchmark_flush(db: InfinityDB) -> dict[str, Any]:
    """Benchmark flush to disk."""
    logger.info("=== FLUSH BENCHMARK ===")
    t0 = time.perf_counter()
    await db.flush()
    total_time = time.perf_counter() - t0
    results = {"flush_seconds": round(total_time, 2)}
    logger.info("  FLUSH DONE: %.2fs", total_time)
    return results


async def benchmark_search(
    db: InfinityDB,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Benchmark vector search at various k values."""
    logger.info("=== SEARCH BENCHMARK: %d iterations per k ===", SEARCH_ITERATIONS)
    results: dict[str, Any] = {}

    for k in SEARCH_K_VALUES:
        latencies: list[float] = []
        for _ in range(SEARCH_ITERATIONS):
            query = rng.standard_normal(DIMENSIONS).astype(np.float32)
            query /= np.linalg.norm(query)

            t0 = time.perf_counter()
            hits = await db.search_similar(query, k=k)
            t1 = time.perf_counter()

            latencies.append(t1 - t0)
            assert len(hits) == k, f"Expected {k} results, got {len(hits)}"

        results[f"k{k}"] = {
            "p50_ms": round(_percentile(latencies, 50) * 1000, 2),
            "p95_ms": round(_percentile(latencies, 95) * 1000, 2),
            "p99_ms": round(_percentile(latencies, 99) * 1000, 2),
            "mean_ms": round(sum(latencies) / len(latencies) * 1000, 2),
        }
        logger.info(
            "  k=%d: p50=%.1fms, p95=%.1fms, p99=%.1fms",
            k, results[f"k{k}"]["p50_ms"], results[f"k{k}"]["p95_ms"],
            results[f"k{k}"]["p99_ms"],
        )

    return results


async def benchmark_bfs(
    db: InfinityDB,
    neuron_count: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Benchmark BFS graph traversal at various depths."""
    logger.info("=== BFS BENCHMARK ===")
    results: dict[str, Any] = {}

    for depth in BFS_DEPTHS:
        latencies: list[float] = []
        node_counts: list[int] = []

        for _ in range(50):  # 50 random starting points
            start_id = f"n{rng.integers(0, neuron_count):08d}"
            t0 = time.perf_counter()
            traversal = await db.bfs_traverse(
                start_id, max_depth=depth, direction="both", max_nodes=500
            )
            t1 = time.perf_counter()

            latencies.append(t1 - t0)
            node_counts.append(len(traversal))

        results[f"depth_{depth}"] = {
            "p50_ms": round(_percentile(latencies, 50) * 1000, 2),
            "p95_ms": round(_percentile(latencies, 95) * 1000, 2),
            "avg_nodes_reached": round(sum(node_counts) / len(node_counts), 1),
        }
        logger.info(
            "  depth=%d: p50=%.1fms, p95=%.1fms, avg_nodes=%.0f",
            depth, results[f"depth_{depth}"]["p50_ms"],
            results[f"depth_{depth}"]["p95_ms"],
            results[f"depth_{depth}"]["avg_nodes_reached"],
        )

    return results


async def benchmark_query(
    db: InfinityDB,
    neuron_count: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Benchmark multi-dimensional RRF queries."""
    logger.info("=== MULTI-DIM QUERY BENCHMARK ===")

    # Query types to benchmark
    scenarios: dict[str, QueryPlan] = {}

    # Vector-only
    vec = rng.standard_normal(DIMENSIONS).astype(np.float32)
    vec /= np.linalg.norm(vec)
    scenarios["vector_only"] = QueryPlan(
        query_vector=vec, vector_weight=1.0, limit=20,
    )

    # Vector + type filter
    scenarios["vector_plus_type"] = QueryPlan(
        query_vector=vec, vector_weight=1.0,
        neuron_type="fact", limit=20,
    )

    # Vector + graph proximity
    seed_idx = rng.integers(0, neuron_count)
    scenarios["vector_plus_graph"] = QueryPlan(
        query_vector=vec, vector_weight=1.0,
        graph_seed_ids=[f"n{seed_idx:08d}"],
        graph_weight=0.5, limit=20,
    )

    # Vector + priority + graph (full multi-dim)
    scenarios["full_multidim"] = QueryPlan(
        query_vector=vec, vector_weight=1.0,
        graph_seed_ids=[f"n{seed_idx:08d}"],
        graph_weight=0.3,
        priority_weight=0.2,
        recency_weight=0.1,
        limit=20,
    )

    # Metadata-only
    scenarios["metadata_only"] = QueryPlan(
        neuron_type="decision",
        content_contains="random",
        limit=20,
    )

    results: dict[str, Any] = {}
    for name, plan in scenarios.items():
        latencies: list[float] = []
        for _ in range(50):
            # Regenerate vector for each iteration
            if plan.query_vector is not None:
                new_vec = rng.standard_normal(DIMENSIONS).astype(np.float32)
                new_vec /= np.linalg.norm(new_vec)
                # QueryPlan is frozen, create new one
                plan_dict = {
                    "query_vector": new_vec,
                    "vector_weight": plan.vector_weight,
                    "neuron_type": plan.neuron_type,
                    "content_contains": plan.content_contains,
                    "graph_seed_ids": plan.graph_seed_ids,
                    "graph_weight": plan.graph_weight,
                    "priority_weight": plan.priority_weight,
                    "recency_weight": plan.recency_weight,
                    "limit": plan.limit,
                }
                # Filter out None/default values
                current_plan = QueryPlan(**{k: v for k, v in plan_dict.items() if v is not None and v != 0.0 and v != []})
            else:
                current_plan = plan

            t0 = time.perf_counter()
            await db.query(current_plan)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        results[name] = {
            "p50_ms": round(_percentile(latencies, 50) * 1000, 2),
            "p95_ms": round(_percentile(latencies, 95) * 1000, 2),
            "p99_ms": round(_percentile(latencies, 99) * 1000, 2),
        }
        logger.info(
            "  %s: p50=%.1fms, p95=%.1fms, p99=%.1fms",
            name, results[name]["p50_ms"], results[name]["p95_ms"],
            results[name]["p99_ms"],
        )

    return results


async def benchmark_reopen(db_dir: Path, brain_id: str) -> dict[str, Any]:
    """Benchmark cold open (load from disk)."""
    logger.info("=== REOPEN BENCHMARK ===")

    gc.collect()
    t0 = time.perf_counter()
    db = InfinityDB(db_dir, brain_id=brain_id, dimensions=DIMENSIONS)
    await db.open()
    t1 = time.perf_counter()

    results = {
        "open_seconds": round(t1 - t0, 2),
        "neuron_count": db.neuron_count,
        "synapse_count": db.synapse_count,
        "fiber_count": db.fiber_count,
        "rss_mb_after_open": round(_get_rss_mb(), 1),
    }
    logger.info("  REOPEN DONE: %.2fs, %d neurons loaded", t1 - t0, db.neuron_count)
    await db.close()
    return results


async def run_benchmark(neuron_count: int, base_dir: Path) -> dict[str, Any]:
    """Run full benchmark suite for a given neuron count."""
    logger.info("\n" + "=" * 60)
    logger.info("STARTING BENCHMARK: %d neurons", neuron_count)
    logger.info("=" * 60)

    brain_id = f"bench-{neuron_count // 1000}k"
    db_dir = base_dir / "brains"
    rng = np.random.default_rng(42)

    db = InfinityDB(db_dir, brain_id=brain_id, dimensions=DIMENSIONS)
    await db.open()

    all_results: dict[str, Any] = {
        "neuron_count": neuron_count,
        "dimensions": DIMENSIONS,
        "batch_size": BATCH_SIZE,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Phase 1: Insert neurons
        all_results["insert"] = await benchmark_insert(db, neuron_count, rng)

        # Phase 2: Insert synapses
        all_results["synapses"] = await benchmark_synapses(db, neuron_count, rng)

        # Phase 3: Create fibers
        all_results["fibers"] = await benchmark_fibers(db, neuron_count, rng)

        # Phase 4: Flush to disk
        all_results["flush"] = await benchmark_flush(db)

        # Disk stats after flush
        disk_mb = _dir_size_mb(db_dir)
        raw_mb = (neuron_count * DIMENSIONS * 4) / 1024 / 1024  # float32 vectors alone
        all_results["disk"] = {
            "total_mb": round(disk_mb, 1),
            "raw_vectors_mb": round(raw_mb, 1),
            "compression_ratio": round(raw_mb / disk_mb, 2) if disk_mb > 0 else 0,
        }
        logger.info("  DISK: %.1f MB total (raw vectors alone: %.1f MB)", disk_mb, raw_mb)

        # Phase 5: Vector search
        all_results["search"] = await benchmark_search(db, rng)

        # Phase 6: BFS traversal
        all_results["bfs"] = await benchmark_bfs(db, neuron_count, rng)

        # Phase 7: Multi-dimensional query
        all_results["query"] = await benchmark_query(db, neuron_count, rng)

        # Phase 8: Close and reopen (cold start)
        await db.close()
        all_results["reopen"] = await benchmark_reopen(db_dir, brain_id)

    except Exception as e:
        logger.error("BENCHMARK FAILED: %s", e, exc_info=True)
        all_results["error"] = str(e)
    else:
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK COMPLETE: %d neurons", neuron_count)
        logger.info("=" * 60)
    finally:
        if db.is_open:
            await db.close()

    all_results["finished_at"] = datetime.now(timezone.utc).isoformat()
    all_results["rss_mb_final"] = round(_get_rss_mb(), 1)
    return all_results


async def main() -> None:
    """Run benchmarks for 1M and 2M neurons."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_dir = Path(__file__).parent / "data"
    base_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {
        "benchmark_version": "1.0",
        "timestamp": timestamp,
        "python_version": sys.version,
        "platform": sys.platform,
    }

    # Run 1M benchmark
    logger.info("\n\n>>> STARTING 1M NEURON BENCHMARK <<<\n")
    all_results["1M"] = await run_benchmark(1_000_000, base_dir / "1M")

    # Force GC between benchmarks
    gc.collect()

    # Run 2M benchmark
    logger.info("\n\n>>> STARTING 2M NEURON BENCHMARK <<<\n")
    all_results["2M"] = await run_benchmark(2_000_000, base_dir / "2M")

    # Save results
    output_path = results_dir / f"stress_test_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("ALL BENCHMARKS COMPLETE")
    logger.info("Results saved to: %s", output_path)
    logger.info("=" * 60)

    # Print summary
    for scale in ["1M", "2M"]:
        if scale in all_results and "error" not in all_results[scale]:
            r = all_results[scale]
            logger.info("\n--- %s Summary ---", scale)
            logger.info("  Insert: %.0f neurons/sec", r["insert"]["neurons_per_second"])
            logger.info("  Disk: %.1f MB", r["disk"]["total_mb"])
            if "search" in r:
                logger.info("  Search k=10: p50=%.1fms, p95=%.1fms",
                            r["search"]["k10"]["p50_ms"], r["search"]["k10"]["p95_ms"])
                logger.info("  Search k=100: p50=%.1fms, p95=%.1fms",
                            r["search"]["k100"]["p50_ms"], r["search"]["k100"]["p95_ms"])
            if "query" in r:
                logger.info("  Multi-dim query: p50=%.1fms, p95=%.1fms",
                            r["query"]["full_multidim"]["p50_ms"],
                            r["query"]["full_multidim"]["p95_ms"])
            if "reopen" in r:
                logger.info("  Cold open: %.2fs", r["reopen"]["open_seconds"])


if __name__ == "__main__":
    asyncio.run(main())
