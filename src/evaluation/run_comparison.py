"""Step 5 orchestrator: Random Forest/MLP (no graph) vs. GCN/GraphSAGE (each graph type x each
sparsification method x each retention level), >=5 seeds per configuration, MAE/RMSE/R2 +
retained nodes/edges + training time + memory.

Fixed test set across every configuration
------------------------------------------
src/baseline/black_hole_sparsification.py derives its own "fixed_test_nodes" internally, from
each graph's own Louvain communities (a handful of high-gravity nodes per community) — meaning
the protected test set differs by graph structure. That's fine for Step 1, which only ever
compares variants of *one* selected graph. Steps 3-5 compare three different graph types against
each other, so the evaluation set must be identical across all of them, or the MAE/RMSE numbers
aren't comparable.

This module therefore computes ONE global fixed test set up front (`_global_fixed_test_nodes`,
a plain seeded random split of node indices, independent of any graph), and:
  - passes it directly to random_sparsify/k_center_sparsify (both accept fixed_test_nodes and
    guarantee those nodes are never dropped);
  - for Black Hole, calls the ported apply_black_hole_sparsification as-is (never modified), then
    `_restore_test_nodes` adds back any of the global test nodes it pruned away, with their
    original edges — a post-processing step in this new module, not a change to the ported code.

This is a real methodological choice, not just an engineering detail — worth the thesis author's
own sign-off before running the full sweep for the final report, since it changes what "Black
Hole" numbers mean relative to how Step 1 reported them.
"""

from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.baseline.black_hole_sparsification import apply_black_hole_sparsification
from src.baseline.gnn_training import (
    GCN,
    GraphSAGE,
    build_pyg_data,
    evaluate_model,
    graph_to_edge_arrays,
    make_splits,
    train_model,
)
from src.common.config import RuntimeConfig
from src.common.logging_setup import get_logger
from src.evaluation.efficiency import graph_efficiency_row, track_efficiency
from src.evaluation.metrics import build_comparison_table
from src.models.mlp import train_and_evaluate_mlp
from src.sparsification_baselines.k_center_sampling import k_center_sparsify
from src.sparsification_baselines.random_sampling import random_sparsify

logger = get_logger(__name__)

GNN_MODEL_CLASSES = {"gcn": GCN, "graphsage": GraphSAGE}


def compute_global_fixed_test_nodes(n: int, config: RuntimeConfig) -> set[int]:
    rng = np.random.default_rng(config.split_random_state)
    n_test = max(1, round(config.test_frac * n))
    return set(rng.choice(n, size=n_test, replace=False).tolist())


def restore_test_nodes(pruned_graph: nx.Graph, original_graph: nx.Graph, fixed_test_nodes: set[int]) -> nx.Graph:
    """Add back any of `fixed_test_nodes` that pruning dropped, with their original edges, so
    every sparsification method leaves an identical, complete evaluation set behind."""
    missing = fixed_test_nodes - set(pruned_graph.nodes())
    if not missing:
        return pruned_graph

    restored = pruned_graph.copy()
    restored.add_nodes_from(missing)
    for node in missing:
        for neighbor, data in original_graph.adj[node].items():
            if neighbor in restored:
                restored.add_edge(node, neighbor, **data)
    return restored


def sparsify_graph(
    graph: nx.Graph,
    df: pd.DataFrame,
    method: str,
    retention: float,
    seed: int,
    fixed_test_nodes: set[int],
    features: np.ndarray,
    config: RuntimeConfig,
) -> dict:
    if method == "black_hole":
        weights = (
            config.gravity_degree_weight,
            config.gravity_betweenness_weight,
            config.gravity_edge_weight_sum_weight,
        )
        result = apply_black_hole_sparsification(graph, df, weights, threshold=1 - retention)
        result["graph"] = restore_test_nodes(result["graph"], graph, fixed_test_nodes)
        return result
    if method == "random":
        return random_sparsify(graph, retention, seed, fixed_test_nodes)
    if method == "k_center":
        return k_center_sparsify(graph, features, retention, seed, fixed_test_nodes)
    raise ValueError(f"Unknown sparsification method {method!r}")


def run_gnn_configuration(
    df: pd.DataFrame,
    features: np.ndarray,
    graph: nx.Graph,
    architecture: str,
    fixed_test_nodes: set[int],
    seed: int,
) -> dict:
    n = len(df)
    labels = df["pld_value"].values.astype(np.float32)

    edge_index, edge_weight = graph_to_edge_arrays(graph, n)
    train_idx, val_idx, test_idx = make_splits(n, fixed_test_nodes, seed=seed)
    data = build_pyg_data(features, edge_index, edge_weight, labels, train_idx, val_idx, test_idx, task="regression")

    torch.manual_seed(seed)  # model weight init is otherwise unseeded, undermining the multi-seed comparison
    model_cls = GNN_MODEL_CLASSES[architecture]
    model = model_cls(features.shape[1], 64, 1)

    with track_efficiency() as stats:
        model, _history = train_model(model, data, task="regression")
        metrics = evaluate_model(model, data, task="regression")

    metrics.update({"seed": seed, "elapsed_seconds": stats.elapsed_seconds, "peak_memory_mb": stats.peak_memory_mb})
    return metrics


def run_non_graph_configuration(
    model_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    fixed_test_nodes: set[int],
    seed: int,
) -> dict:
    n = len(features)
    train_idx, val_idx, test_idx = make_splits(n, fixed_test_nodes, seed=seed)

    with track_efficiency() as stats:
        if model_name == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            model = RandomForestRegressor(n_estimators=100, random_state=seed)
            model.fit(features[train_idx], labels[train_idx])
            y_pred = model.predict(features[test_idx])
            metrics = {
                "mae": float(mean_absolute_error(labels[test_idx], y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(labels[test_idx], y_pred))),
                "r2": float(r2_score(labels[test_idx], y_pred)),
                "seed": seed,
            }
        elif model_name == "mlp":
            metrics = train_and_evaluate_mlp(
                features[train_idx], labels[train_idx],
                features[val_idx], labels[val_idx],
                features[test_idx], labels[test_idx],
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown non-graph baseline {model_name!r}")

    metrics.update({"elapsed_seconds": stats.elapsed_seconds, "peak_memory_mb": stats.peak_memory_mb})
    return metrics


def run_full_comparison(
    df: pd.DataFrame,
    features: np.ndarray,
    graphs: dict[str, nx.Graph],
    config: RuntimeConfig,
) -> pd.DataFrame:
    """Runs every (model x graph type x sparsification method x retention level) configuration
    for config.seeds seeds each, and returns the aggregated mean +/- std comparison table.

    `graphs` maps graph_type -> nx.Graph, e.g. built via src.graphs.fingerprint_graph,
    src.graphs.semantic_graph, src.graphs.hybrid_graph (see scripts/run_step3_build_graphs.py).
    """
    n = len(df)
    labels = df["pld_value"].values.astype(np.float32)
    fixed_test_nodes = compute_global_fixed_test_nodes(n, config)

    configurations: dict[tuple, list[dict]] = {}
    retained_structure: dict[tuple, dict] = {}

    for model_name in config.non_graph_baselines:
        key = (model_name, "none", "none", 1.0)
        configurations[key] = [
            run_non_graph_configuration(model_name, features, labels, fixed_test_nodes, seed)
            for seed in config.seeds
        ]

    for graph_type, graph in graphs.items():
        for method in config.sparsification_methods:
            for retention in config.retention_levels:
                per_seed_metrics = []
                for seed in config.seeds:
                    sparsified = sparsify_graph(graph, df, method, retention, seed, fixed_test_nodes, features, config)
                    pruned_graph = sparsified["graph"]

                    for architecture in config.gnn_architectures:
                        arch_key = (architecture, graph_type, method, retention)
                        metrics = run_gnn_configuration(df, features, pruned_graph, architecture, fixed_test_nodes, seed)
                        configurations.setdefault(arch_key, []).append(metrics)
                        retained_structure[arch_key] = graph_efficiency_row(pruned_graph)

                logger.info(f"Completed {graph_type}/{method}/retention={retention}")

    table = build_comparison_table(configurations)
    retained_nodes, retained_edges = [], []
    for _, row in table.iterrows():
        key = (row["model"], row["graph_type"], row["sparsification_method"], row["retention_level"])
        structure = retained_structure.get(key, {})
        retained_nodes.append(structure.get("retained_nodes"))
        retained_edges.append(structure.get("retained_edges"))
    table["retained_nodes"] = retained_nodes
    table["retained_edges"] = retained_edges
    return table
