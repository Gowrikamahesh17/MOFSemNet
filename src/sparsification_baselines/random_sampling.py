"""Random sampling baseline for Step 4 — the simplest possible comparison point for Black Hole
sparsification (src/baseline/black_hole_sparsification.py): keep a uniform-random `retention`
fraction of nodes, with no structural reasoning at all. If Black Hole doesn't beat this, its
gravity-based pruning isn't earning its complexity.
"""

from __future__ import annotations

import time

import networkx as nx
import numpy as np

from src.common.logging_setup import get_logger

logger = get_logger(__name__)


def random_sparsify(graph: nx.Graph, retention: float, seed: int, fixed_test_nodes: set | None = None) -> dict:
    """Keep round(retention * n) nodes uniformly at random, always including
    `fixed_test_nodes` (the test set must never be pruned away — matched to Black Hole's own
    guarantee in src/baseline/black_hole_sparsification.py::black_hole_strategy_per_community).

    Returns the same shape as apply_black_hole_sparsification: {"graph", "metrics"}, so
    src.evaluation.run_comparison can treat all three sparsification methods interchangeably.
    """
    if not 0.0 < retention <= 1.0:
        raise ValueError(f"retention must be in (0, 1], got {retention}")

    fixed_test_nodes = fixed_test_nodes or set()
    start = time.time()

    nodes = list(graph.nodes())
    n_before, e_before = graph.number_of_nodes(), graph.number_of_edges()
    n_to_keep = max(len(fixed_test_nodes), round(retention * n_before))

    rng = np.random.default_rng(seed)
    candidates = [node for node in nodes if node not in fixed_test_nodes]
    n_extra = max(0, n_to_keep - len(fixed_test_nodes))
    sampled_extra = rng.choice(candidates, size=min(n_extra, len(candidates)), replace=False).tolist()

    nodes_to_keep = set(fixed_test_nodes) | set(sampled_extra)
    pruned_graph = graph.subgraph(nodes_to_keep).copy()

    n_after, e_after = pruned_graph.number_of_nodes(), pruned_graph.number_of_edges()
    metrics = {
        "method": "random",
        "seed": seed,
        "nodes_before": n_before,
        "edges_before": e_before,
        "nodes_after": n_after,
        "edges_after": e_after,
        "node_retention_pct": round(100 * n_after / n_before, 2) if n_before else 0.0,
        "edge_retention_pct": round(100 * e_after / e_before, 2) if e_before else 0.0,
        "elapsed_seconds": round(time.time() - start, 4),
    }
    logger.info(f"Random sparsification (retention={retention}, seed={seed}): {metrics}")
    return {"graph": pruned_graph, "metrics": metrics}
