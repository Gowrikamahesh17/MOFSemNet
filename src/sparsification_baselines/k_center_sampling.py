"""Greedy k-center (farthest-point) coreset sampling baseline for Step 4.

No suitable existing package was worth adding for this (the well-known scikit-learn-compatible
coreset libraries are either unmaintained or pull in dependencies well beyond this project's
scope for one sampling strategy), so it's implemented directly: the standard greedy 2-approximate
k-center algorithm, operating on node *feature* vectors (not graph structure) — this is
deliberately structure-agnostic, since the point of comparing it to Black Hole is to see whether
Black Hole's graph-aware gravity pruning beats a baseline that only knows about feature-space
diversity.
"""

from __future__ import annotations

import time

import networkx as nx
import numpy as np

from src.common.logging_setup import get_logger

logger = get_logger(__name__)


def _greedy_k_center_indices(features: np.ndarray, n_centers: int, seed: int, required: list[int]) -> list[int]:
    """Greedily grow a center set that maximizes the minimum distance to any already-chosen
    center, starting from `required` (e.g. the fixed test nodes) so they're always kept, then
    filling the remaining budget with the standard farthest-point heuristic."""
    n = features.shape[0]
    rng = np.random.default_rng(seed)

    if not required:
        required = [int(rng.integers(0, n))]

    selected = list(dict.fromkeys(required))  # required, de-duplicated, order preserved
    min_dist = np.full(n, np.inf)
    for center in selected:
        dist = np.linalg.norm(features - features[center], axis=1)
        min_dist = np.minimum(min_dist, dist)

    while len(selected) < n_centers and len(selected) < n:
        next_center = int(np.argmax(min_dist))
        if min_dist[next_center] == 0:
            break  # every remaining point coincides with an already-selected center
        selected.append(next_center)
        dist = np.linalg.norm(features - features[next_center], axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[next_center] = 0.0

    return selected[:n_centers]


def k_center_sparsify(
    graph: nx.Graph,
    features: np.ndarray,
    retention: float,
    seed: int,
    fixed_test_nodes: set | None = None,
) -> dict:
    """Keep round(retention * n) nodes chosen by greedy k-center over `features` (row-aligned
    with graph node order — pass whichever node representation is being sparsified: fingerprint,
    semantic embedding, or hybrid concatenation).

    Returns the same {"graph", "metrics"} shape as random_sparsify and
    apply_black_hole_sparsification.
    """
    if not 0.0 < retention <= 1.0:
        raise ValueError(f"retention must be in (0, 1], got {retention}")
    if features.shape[0] != graph.number_of_nodes():
        raise ValueError(
            f"features has {features.shape[0]} rows but graph has {graph.number_of_nodes()} nodes; "
            "features must be row-aligned with graph node indices."
        )

    fixed_test_nodes = fixed_test_nodes or set()
    start = time.time()

    n_before, e_before = graph.number_of_nodes(), graph.number_of_edges()
    n_to_keep = max(len(fixed_test_nodes), round(retention * n_before))

    selected = _greedy_k_center_indices(features, n_to_keep, seed, sorted(fixed_test_nodes))
    node_list = list(graph.nodes())
    nodes_to_keep = {node_list[i] for i in selected}

    pruned_graph = graph.subgraph(nodes_to_keep).copy()

    n_after, e_after = pruned_graph.number_of_nodes(), pruned_graph.number_of_edges()
    metrics = {
        "method": "k_center",
        "seed": seed,
        "nodes_before": n_before,
        "edges_before": e_before,
        "nodes_after": n_after,
        "edges_after": e_after,
        "node_retention_pct": round(100 * n_after / n_before, 2) if n_before else 0.0,
        "edge_retention_pct": round(100 * e_after / e_before, 2) if e_before else 0.0,
        "elapsed_seconds": round(time.time() - start, 4),
    }
    logger.info(f"k-center sparsification (retention={retention}, seed={seed}): {metrics}")
    return {"graph": pruned_graph, "metrics": metrics}
