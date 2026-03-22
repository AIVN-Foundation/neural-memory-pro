# Neural Memory Pro

Advanced extensions for [Neural Memory](https://github.com/nhadaututtheky/neural-memory) — the persistent memory system for AI agents.

## Features

- **Cone Queries** — Exhaustive recall via embedding similarity cones. Never miss a relevant memory.
- **Directional Compression** — Multi-axis semantic compression preserving relationships to multiple concepts.
- **Smart Merge Consolidation** — Priority-aware clustering with temporal coherence for intelligent memory merging.

## Installation

```bash
pip install git+https://github.com/AIVN-Foundation/neural-memory-pro.git
```

> Requires a valid purchase. Get access at [theio.vn](https://theio.vn).

## Usage

Once installed, Pro features are automatically available in Neural Memory — no configuration needed.

```python
# Pro features are auto-discovered via entry_points
from neural_memory.plugins import has_pro

if has_pro():
    print("Pro features active!")
```

## License

Proprietary — © AIVN Foundation. All rights reserved.
