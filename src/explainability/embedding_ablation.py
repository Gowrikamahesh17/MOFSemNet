"""Step 6b: embedding ablation for the language representations.

Question: when a graph or a feature set built from language-model embeddings scores a certain R2,
how much of that comes from real chemical information in the embeddings, versus incidental structure
(for example that many MOFs share an identical linker, which makes them identical in any embedding)?

Method: degrade the embeddings in a controlled way, rebuild the semantic graph (or the input
features) from the degraded version, re-run the same models and compare R2 with the full-embedding
result. If a graph built from random numbers scores the same as one built from ChemBERTa, the
embedding contributes no usable chemistry to that result.

Ablations act on the DISTINCT linker embeddings (MOFs that share a linker keep sharing one vector),
so the cluster structure of identical linkers is preserved in every variant and only the content changes:
  "none"               full embeddings (reference)
  "shuffle_rows"       real vectors assigned to the wrong linkers (right distribution, wrong chemistry)
  "random_replacement" Gaussian noise vectors, one per distinct linker
  "truncate_dims"      keep only the first `keep_dims` dimensions
(Shuffling the dimensions of every vector the same way would not change cosine similarity, so it is
not a useful ablation and is not offered.)
"""

from __future__ import annotations

import time

import numpy as np
import torch

ABLATION_TYPES = ("none", "shuffle_rows", "random_replacement", "truncate_dims")


def ablate_embeddings(embeddings: np.ndarray, ablation_type: str, seed: int = 0, keep_dims: int = 64) -> np.ndarray:
    """Return an ablated copy of `embeddings` (same number of rows; columns change only for truncate_dims)."""
    if ablation_type not in ABLATION_TYPES:
        raise ValueError(f"Unknown ablation_type {ablation_type!r}; choose from {ABLATION_TYPES}")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if ablation_type == "none":
        return embeddings.copy()
    if ablation_type == "truncate_dims":
        if not 1 <= keep_dims <= embeddings.shape[1]:
            raise ValueError("keep_dims must be between 1 and the embedding dimension")
        return np.ascontiguousarray(embeddings[:, :keep_dims])

    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(embeddings, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    if ablation_type == "shuffle_rows":
        unique_new = unique[rng.permutation(len(unique))]
    else:  # random_replacement: same mean / spread per dimension as the real vectors
        unique_new = rng.normal(unique.mean(axis=0), unique.std(axis=0) + 1e-8, size=unique.shape).astype(np.float32)
    return unique_new[inverse]


def _gnn_r2(model_name, X, graph, labels, test_nodes, seed):
    from src.baseline.gnn_training import (GCN, GraphSAGE, build_pyg_data, evaluate_model, graph_to_edge_arrays,
                                           make_splits, train_model)
    n = len(labels)
    edge_index, edge_weight = graph_to_edge_arrays(graph, n)
    train_idx, val_idx, test_idx = make_splits(n, test_nodes, seed=seed)
    data = build_pyg_data(X, edge_index, edge_weight, labels, train_idx, val_idx, test_idx, task="regression")
    torch.manual_seed(seed)
    model = {"gcn": GCN, "graphsage": GraphSAGE}[model_name](X.shape[1], 64, 1)
    model, _ = train_model(model, data, task="regression")
    return evaluate_model(model, data, task="regression")


def run_embedding_ablation(embeddings: np.ndarray, ablation_type: str, config, *, X: np.ndarray, labels: np.ndarray,
                           test_nodes: set, k: int = 3, models=("gcn", "graphsage"), seeds=None,
                           keep_dims: int = 64) -> list[dict]:
    """Rebuild the semantic graph (src.graphs.semantic_graph) from an ablated version of `embeddings`
    and train/evaluate the GNNs on node features `X` (fingerprint features) over that graph.
    One dict per (model, seed): ablation, model, seed, mae, rmse, r2, n_edges, elapsed_seconds.
    The ablation itself is seeded with each run's seed, so shuffles/noise differ between seeds."""
    from src.graphs.semantic_graph import build_semantic_graph

    rows = []
    for seed in (seeds if seeds is not None else config.seeds):
        ablated = ablate_embeddings(embeddings, ablation_type, seed=seed, keep_dims=keep_dims)
        graph = build_semantic_graph(ablated, k=k)
        for model_name in models:
            t = time.time()
            m = _gnn_r2(model_name, X, graph, labels, test_nodes, seed)
            rows.append({"ablation": ablation_type, "model": model_name, "k": k, "seed": seed, "mae": m["mae"],
                         "rmse": m["rmse"], "r2": m["r2"], "n_edges": graph.number_of_edges(),
                         "elapsed_seconds": time.time() - t})
    return rows
