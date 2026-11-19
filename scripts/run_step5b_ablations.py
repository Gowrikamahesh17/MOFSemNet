#!/usr/bin/env python
"""Step 5b — diagnostic experiments run after the first Step 5 sweep (see
reports/logs/2026-11-18-step5-chemberta-sweep-and-step1-gap.md).

Question behind each block (numbers match the discussion with the supervisor):
  1. no_edge        GNN with NO edges (what does the graph add or remove?)
  2. cs2_graph      Case Study 2's graph (0.9 Tanimoto + 0.1 same-metal, kNN k=3), full and
                    Black Hole 30% / 50%, on the Step 5 test set
  3. small_k        fingerprint / semantic / hybrid graphs rebuilt with k=3 instead of k=10
  4. embeddings     ChemBERTa as INPUT features: RF / MLP / no-edge GNN on fingerprint,
                    embedding, and fingerprint+embedding
  5. residual       ResidualGCN (starts as an MLP, graph gated by a learnable weight)
  plus the unpruned ("full") versions of the three Step 5 graphs, as the reference row.

Everything uses the same fixed test set as Step 5 (config.split_random_state=42) and seeds 0-4.
Writes per-seed rows and an aggregated mean/std table to outputs/results/.

Usage:
    PYTHONPATH=. python scripts/run_step5b_ablations.py --model chemberta
"""

import argparse
import hashlib
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.baseline.black_hole_sparsification import apply_black_hole_sparsification
from src.baseline.data_ingestion import load_dataset
from src.baseline.feature_engineering import build_features
from src.baseline.gnn_training import (
    GCN, GraphSAGE, build_pyg_data, evaluate_model, graph_to_edge_arrays, make_splits, train_model,
)
from src.baseline.graph_construction import build_similarity_graphs
from src.common.config import LANGUAGE_MODEL_CHOICES, RuntimeConfig
from src.common.logging_setup import get_logger
from src.evaluation.run_comparison import compute_global_fixed_test_nodes, restore_test_nodes
from src.graphs.fingerprint_graph import build_fingerprint_graph
from src.graphs.hybrid_graph import build_hybrid_graph
from src.graphs.semantic_graph import build_semantic_graph
from src.language.embeddings import cached_embed
from src.models.gnn_residual import ResidualGCN
from src.models.mlp import train_and_evaluate_mlp

logger = get_logger(__name__)

SCRIPT = "scripts/run_step5b_ablations.py"
GNN_CLASSES = {"gcn": GCN, "graphsage": GraphSAGE, "residual_gcn": ResidualGCN}


def run_gnn(model_name, X, graph, df, labels, test_nodes, seed):
    n = len(df)
    edge_index, edge_weight = graph_to_edge_arrays(graph, n)
    train_idx, val_idx, test_idx = make_splits(n, test_nodes, seed=seed)
    data = build_pyg_data(X, edge_index, edge_weight, labels, train_idx, val_idx, test_idx, task="regression")
    torch.manual_seed(seed)
    model = GNN_CLASSES[model_name](X.shape[1], 64, 1)
    t = time.time()
    model, _ = train_model(model, data, task="regression")
    m = evaluate_model(model, data, task="regression")
    m["elapsed_seconds"] = time.time() - t
    return m


def run_rf(X, labels, test_nodes, seed):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    tr, _va, te = make_splits(len(X), test_nodes, seed=seed)
    t = time.time()
    p = RandomForestRegressor(n_estimators=100, random_state=seed).fit(X[tr], labels[tr]).predict(X[te])
    return {"mae": float(mean_absolute_error(labels[te], p)),
            "rmse": float(np.sqrt(mean_squared_error(labels[te], p))),
            "r2": float(r2_score(labels[te], p)), "elapsed_seconds": time.time() - t}


def run_mlp(X, labels, test_nodes, seed):
    tr, va, te = make_splits(len(X), test_nodes, seed=seed)
    t = time.time()
    m = train_and_evaluate_mlp(X[tr], labels[tr], X[va], labels[va], X[te], labels[te], seed=seed)
    m["elapsed_seconds"] = time.time() - t
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=LANGUAGE_MODEL_CHOICES, default="chemberta")
    args = parser.parse_args()

    config = RuntimeConfig(language_model=args.model)
    df = load_dataset("large")
    labels = df["pld_value"].values.astype(np.float32)
    X, _ = build_features(df, feature_scheme="fingerprint")
    E = cached_embed(df["linker_smiles"].tolist(), config).astype(np.float32)
    XE = np.hstack([X, E]).astype(np.float32)
    test_nodes = compute_global_fixed_test_nodes(len(df), config)
    data_sha = hashlib.sha256(open(config.dataset_path, "rb").read()).hexdigest()[:16]
    logger.info(f"X={X.shape} E={E.shape} test={len(test_nodes)} data_sha256[:16]={data_sha}")

    n = len(df)
    empty = nx.Graph(); empty.add_nodes_from(range(n))
    smiles = df["linker_smiles"]
    graphs = {
        "none": empty,
        "cs2_knn3": build_similarity_graphs(df, "large")["knn_3"],
        "fingerprint_k10": build_fingerprint_graph(smiles, k=10),
        "semantic_k10": build_semantic_graph(E, k=10),
        "hybrid_k10": build_hybrid_graph(smiles, E, k=10),
        "fingerprint_k3": build_fingerprint_graph(smiles, k=3),
        "semantic_k3": build_semantic_graph(E, k=3),
        "hybrid_k3": build_hybrid_graph(smiles, E, k=3),
    }
    weights = (config.gravity_degree_weight, config.gravity_betweenness_weight, config.gravity_edge_weight_sum_weight)
    for retention in (0.3, 0.5):
        bh = apply_black_hole_sparsification(graphs["cs2_knn3"], df, weights, threshold=1 - retention)
        graphs[f"cs2_knn3_bh{int(retention * 100)}"] = restore_test_nodes(bh["graph"], graphs["cs2_knn3"], test_nodes)

    os.makedirs(config.results_dir, exist_ok=True)
    per_seed_path = os.path.join(config.results_dir, f"step5b_ablations_{args.model}_per_seed.csv")
    rows = []

    def record(experiment, model, features, graph, seed, metrics):
        rows.append({"experiment": experiment, "model": model, "features": features, "graph": graph,
                     "seed": seed, "mae": metrics["mae"], "rmse": metrics["rmse"], "r2": metrics["r2"],
                     "elapsed_seconds": metrics.get("elapsed_seconds"), "n_test": len(test_nodes),
                     "language_model": args.model, "script": SCRIPT, "data_sha256_16": data_sha,
                     "split_random_state": config.split_random_state})
        pd.DataFrame(rows).to_csv(per_seed_path, index=False)

    # 4. ChemBERTa as input features (non-graph models: fast, run first)
    for feat_name, feats in (("fingerprint", X), ("embedding", E), ("fingerprint+embedding", XE)):
        for seed in config.seeds:
            record("embeddings_as_input", "random_forest", feat_name, "none", seed, run_rf(feats, labels, test_nodes, seed))
            record("embeddings_as_input", "mlp", feat_name, "none", seed, run_mlp(feats, labels, test_nodes, seed))
        logger.info(f"done non-graph models on {feat_name}")

    # 1/2/3 + reference: GCN and GraphSAGE on every graph (fingerprint features)
    exp_of = {"none": "no_edge", "cs2_knn3": "cs2_graph", "cs2_knn3_bh30": "cs2_graph", "cs2_knn3_bh50": "cs2_graph",
              "fingerprint_k10": "full_graph_reference", "semantic_k10": "full_graph_reference",
              "hybrid_k10": "full_graph_reference", "fingerprint_k3": "small_k", "semantic_k3": "small_k",
              "hybrid_k3": "small_k"}
    for gname, g in graphs.items():
        for arch in ("gcn", "graphsage"):
            for seed in config.seeds:
                record(exp_of[gname], arch, "fingerprint", gname, seed, run_gnn(arch, X, g, df, labels, test_nodes, seed))
        logger.info(f"done GCN/GraphSAGE on graph={gname} (edges={g.number_of_edges()})")

    # 4 (cont.): no-edge GNNs on fingerprint+embedding
    for arch in ("gcn", "graphsage"):
        for seed in config.seeds:
            record("embeddings_as_input", arch, "fingerprint+embedding", "none", seed,
                   run_gnn(arch, XE, graphs["none"], df, labels, test_nodes, seed))
    logger.info("done no-edge GNNs on fingerprint+embedding")

    # 5. Residual GCN (starts as an MLP; graph gated)
    for gname in ("none", "cs2_knn3", "fingerprint_k10", "semantic_k10", "hybrid_k10", "semantic_k3", "hybrid_k3"):
        for seed in config.seeds:
            record("residual", "residual_gcn", "fingerprint", gname, seed,
                   run_gnn("residual_gcn", X, graphs[gname], df, labels, test_nodes, seed))
        logger.info(f"done ResidualGCN on graph={gname}")

    per_seed = pd.DataFrame(rows)
    agg = (per_seed.groupby(["experiment", "model", "features", "graph"], sort=False)
           .agg(n_seeds=("seed", "nunique"), mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), r2_mean=("r2", "mean"),
                r2_std=("r2", "std"), seconds_mean=("elapsed_seconds", "mean")).reset_index())
    out = os.path.join(config.results_dir, f"step5b_ablations_{args.model}.csv")
    agg.to_csv(out, index=False)
    logger.info(f"Saved {out}")
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
