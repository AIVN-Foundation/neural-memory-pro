# Changelog

All notable changes to Neural Memory Pro are documented here.

## [0.2.0] - 2026-03-23

### Added
- **InfinityDB Engine** — Custom multi-dimensional spatial database
  - HNSW approximate nearest neighbor search (hnswlib)
  - Memory-mapped vector store (numpy)
  - Graph store with adjacency lists (msgpack)
  - Metadata store with indexed fields
  - Fiber store for memory groupings
  - Write-ahead log (WAL) for crash safety
  - 5-tier compression system (ACTIVE/WARM/COOL/FROZEN/CRYSTAL)
  - Multi-dimensional query planner with RRF fusion
  - SQLite → InfinityDB migration engine
- **NeuralStorage Adapter** — Drop-in replacement for SQLite backend
  - Full NeuralStorage interface (24+ methods)
  - Neuron/Synapse/Fiber CRUD with frozen dataclass converters
  - Graph traversal (get_neighbors, get_path via BFS)
  - Brain export/import (BrainSnapshot serialization)
  - Pro-specific: search_similar, demote_sweep, get_tier_stats
- **Pro MCP Tools** — 3 exclusive tools via plugin system
  - `nmem_cone_query` — HNSW cone recall with similarity threshold
  - `nmem_tier_info` — Tier distribution stats + demote sweep
  - `nmem_pro_merge` — Smart merge with dry_run preview
- **Plugin Storage Hook** — `get_storage_class()` for auto-upgrade
- **Cone Queries** — HNSW-accelerated retrieval (O(N) → O(k))
- **Smart Merge** — HNSW-accelerated consolidation (O(N^2) → O(N*k))
- **Directional Compress** — Multi-axis semantic compression (4 levels)
- **Benchmark Results** — Stress tested at 100K/1M/2M neurons
  - 100K (384D): 1.7K inserts/s, 3.3ms search p50, 424MB
  - 1M (64D): 6.5K inserts/s, 4.0ms search p50, 1.4GB
  - 2M (32D): 4.1K inserts/s, 2.2ms search p50, 2.2GB

### Tests
- 304+ tests across 12 test files
- Full integration tests for WAL, compression, adapter, MCP tools

## [0.1.0] - 2026-03-20

### Added
- Initial scaffold with plugin registration
- Cone queries (brute-force, pre-InfinityDB)
- Directional compression (basic)
- Smart merge (basic)
- Plugin auto-discovery via entry_points
