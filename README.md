# Neural Memory Pro

> Drop-in upgrade for [Neural Memory](https://github.com/nhadaututtheky/neural-memory) — replaces SQLite with a purpose-built spatial database engine.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-proprietary-red.svg)](LICENSE)

## Why Pro?

Neural Memory (free) uses SQLite — great for getting started, but it hits walls at scale:

| Capability | Free (SQLite) | Pro (InfinityDB) |
|-----------|---------------|-------------------|
| **Vector search** | Sequential scan | HNSW index, sub-5ms at 1M neurons |
| **Max neurons** | ~50K practical | 2M+ tested, designed for 10M+ |
| **Compression** | None | 5-tier adaptive (up to 89% ratio) |
| **Storage engine** | Generic relational DB | Purpose-built for neural graphs |
| **Tiered storage** | All in memory | Hot/warm/cold with auto-demotion |
| **Graph traversal** | SQL JOINs | Native adjacency + BFS (<1ms depth-3) |
| **MCP tools** | 52 tools | 52 + 3 Pro-exclusive tools |
| **Recall speed** | ~50ms (small brains) | <5ms p50 at 100K neurons |

## Installation

```bash
# One command — automatically installs neural-memory (free) as dependency
pip install git+https://github.com/AIVN-Foundation/neural-memory-pro.git
```

That's it. No configuration needed — Pro auto-registers via Python entry_points and upgrades the storage backend transparently.

### Verify Installation

```bash
nmem version
# neural-memory 4.19.0 (Pro: InfinityDB 0.2.0)
```

Or in Python:

```python
from neural_memory.plugins import has_pro
print(has_pro())  # True
```

## What You Get

### InfinityDB Engine

A custom spatial database engine built specifically for neural memory graphs:

- **HNSW Vector Index** — Hierarchical Navigable Small World graph for approximate nearest neighbor search. Sub-5ms queries at 1M+ neurons.
- **Write-Ahead Log (WAL)** — Crash-safe writes with automatic recovery on restart.
- **5-Tier Compression** — Adaptive compression pipeline (none → LZ4 → zstd → quantization → cold archive). Automatically selects tier based on access patterns.
- **Tiered Storage** — Hot neurons stay in memory, warm on SSD, cold compressed. Auto-demotion based on access frequency.
- **Native Graph Store** — Adjacency lists stored alongside vectors. BFS traversal in <1ms for depth-3.
- **Query Planner** — Optimizes retrieval strategy based on query type (vector similarity, graph traversal, hybrid).

### Pro MCP Tools

Three additional tools available in Claude Code when Pro is installed:

| Tool | Description |
|------|-------------|
| `nmem_cone_query` | HNSW cone recall — find all memories within a similarity threshold. Never miss a relevant memory. |
| `nmem_tier_info` | Storage tier statistics + trigger demote sweep for cold neurons. |
| `nmem_pro_merge` | Smart merge consolidation with dry-run preview. Priority-aware clustering with temporal coherence. |

### Pro Retrieval Strategies

- **Cone Queries** — Exhaustive recall via embedding similarity cones. Unlike top-k, cone queries guarantee no relevant memory is missed within a similarity threshold.
- **Directional Compression** — Multi-axis semantic compression that preserves relationships to multiple concepts simultaneously. Reduces storage while maintaining recall quality.
- **Smart Merge** — Priority-aware clustering with temporal coherence. Groups related memories and merges them intelligently, respecting priority and recency.

## Benchmarks

Tested on Windows 11, Python 3.14, consumer hardware (no GPU needed):

### Insert Performance

| Scale | Neurons/sec | Total Time |
|-------|------------|------------|
| 100K neurons (384D) | 1,714/s | 58s |
| 1M neurons (64D) | 6,463/s | 2.5min |
| 2M neurons (32D) | 4,119/s | 8min |

### Search Latency (p50)

| Scale | k=10 | k=50 | k=100 |
|-------|------|------|-------|
| 100K | 3.3ms | 8.9ms | 15.3ms |
| 1M | 4.0ms | 13.7ms | 23.8ms |
| 2M | 2.2ms | 8.0ms | 14.6ms |

### Graph Traversal (BFS, p50)

| Scale | Depth 1 | Depth 2 | Depth 3 |
|-------|---------|---------|---------|
| 100K | 0.15ms | 0.20ms | 0.53ms |
| 1M | 0.24ms | 0.36ms | 1.08ms |
| 2M | 0.17ms | 0.26ms | 0.64ms |

### Compression

| Scale | Raw Vectors | On Disk | Ratio |
|-------|------------|---------|-------|
| 100K (384D) | 146 MB | 424 MB | 35% |
| 1M (64D) | 244 MB | 1,358 MB | 18% |
| 2M (32D) | 244 MB | 2,217 MB | 11% |

> Full benchmark data: `benchmarks/results/`

## Architecture

```
neural-memory-pro/
  src/neural_memory_pro/
    infinitydb/
      engine.py          — Core database engine (open, close, CRUD, flush)
      hnsw_index.py      — HNSW vector index (hnswlib wrapper)
      vector_store.py    — Vector storage with batch operations
      graph_store.py     — Native adjacency list graph store
      metadata_store.py  — Neuron/synapse metadata (msgpack serialized)
      fiber_store.py     — Fiber (memory cluster) storage
      compressor.py      — 5-tier adaptive compression
      tier_manager.py    — Hot/warm/cold tiered storage
      wal.py             — Write-ahead log for crash safety
      query_planner.py   — Query optimization and strategy selection
      file_format.py     — Binary file format spec
      migrator.py        — Schema migration between versions
    retrieval/
      cone_queries.py    — Exhaustive similarity cone recall
    consolidation/
      smart_merge.py     — Priority-aware memory merging
    hyperspace/
      directional_compress.py — Multi-axis semantic compression
    storage_adapter.py   — NeuralStorage interface adapter
    mcp_tools.py         — Pro-exclusive MCP tool schemas + handlers
    plugin.py            — Plugin registration (auto-discovered)
```

## How It Works

1. **Install** — `pip install` pulls Pro + free tier as dependency
2. **Auto-register** — Python entry_points system discovers `neural_memory_pro:auto_register`
3. **Storage upgrade** — Free tier's `storage/factory.py` detects Pro plugin → uses InfinityDB instead of SQLite
4. **Transparent** — All 52 free MCP tools work unchanged. 3 Pro tools are added automatically.
5. **Fallback** — If Pro is uninstalled, free tier falls back to SQLite with zero errors

```
┌─────────────────────────────────┐
│  Claude Code / MCP Client       │
├─────────────────────────────────┤
│  Neural Memory (free)           │
│  52 MCP tools, retrieval engine │
├──────────┬──────────────────────┤
│ SQLite   │ InfinityDB (Pro)     │
│ (free)   │ HNSW + WAL + Tiers   │
└──────────┴──────────────────────┘
        ↑ auto-selected based on
          whether Pro is installed
```

## Requirements

- Python 3.11+
- `neural-memory >= 4.18.0` (auto-installed)
- `numpy >= 1.24`
- `hnswlib >= 0.8.0`
- `msgpack >= 1.0`
- No GPU required. Runs on consumer hardware.

## License

Proprietary — AIVN Foundation. All rights reserved.

Get access at [theio.vn](https://theio.vn).
