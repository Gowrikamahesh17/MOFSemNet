#!/usr/bin/env python
"""Step 6c - explainers for GNNs WITH a graph: SHAP and LIME for GCN, GraphSAGE and the residual GCN on the CS2 graph (k=3),
plus LIME for the no-edge GCN. Same data, features, random test set (split state 42) and seed-0 split as Step 6 / 6b.

For each explained test MOF:
  * SHAP (expected gradients) over the features of its whole k-hop neighbourhood -> how much of the explanation sits on the
    MOF's own features vs. its neighbours, and which input groups drive the own-feature part;
  * LIME on the MOF's own features with the neighbours held at their real values, compared with the SHAP own-feature part.

Usage:
    PYTHONPATH=. python scripts/run_step6c_gnn_explainers.py [--n-nodes 30] [--lime-samples 5000]
"""

import argparse
import hashlib
import os
import time

import numpy as np
import pandas as pd

from src.baseline.data_ingestion import load_dataset
from src.baseline.feature_engineering import build_features
from src.baseline.gnn_training import graph_to_edge_arrays, make_splits
from src.common.config import RuntimeConfig
from src.common.logging_setup import get_logger
from src.evaluation.run_comparison import compute_global_fixed_test_nodes
from src.explainability.model_explainers import (HOPS, gcn_predict, graph_gnn_shap, graph_predict_fn, group_shares, lime_one,
                                                 standardise_inputs, train_gcn)
from src.explainability.shap_analysis import feature_group, fingerprint_feature_names

logger = get_logger(__name__)
SCRIPT = "scripts/run_step6c_gnn_explainers.py"
UI_DIR = os.path.join("frontend", "public", "step6")
SEED = 0
MODELS = ("gcn", "graphsage", "residual_gcn")


def r2(y, p):
    from sklearn.metrics import r2_score
    return float(r2_score(y, p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-nodes", type=int, default=30)
    ap.add_argument("--lime-samples", type=int, default=5000)
    ap.add_argument("--shap-samples", type=int, default=50)
    args = ap.parse_args()

    config = RuntimeConfig(language_model="chemberta")
    df = load_dataset("large")
    y = df["pld_value"].values.astype(np.float32)
    X, meta = build_features(df, feature_scheme="fingerprint")
    names = fingerprint_feature_names(meta)
    groups = np.array([feature_group(n) for n in names])
    test_nodes = compute_global_fixed_test_nodes(len(df), config)
    tr, va, _ = make_splits(len(X), test_nodes, seed=SEED)
    te = np.array(sorted(test_nodes))
    non_test = np.array([i for i in range(len(X)) if i not in test_nodes])
    Xs, *_ = standardise_inputs(X, non_test)
    is_bin = ((X == 0) | (X == 1)).all(axis=0)
    cat_idx = list(np.where(is_bin)[0])
    data_sha = hashlib.sha256(open(config.dataset_path, "rb").read()).hexdigest()[:16]
    base = {"script": SCRIPT, "data_sha256_16": data_sha, "split_random_state": config.split_random_state, "seed": SEED}
    rng = np.random.default_rng(SEED)
    nodes = np.sort(rng.choice(te, size=min(args.n_nodes, len(te)), replace=False))
    lime_train = Xs[rng.choice(tr, size=2000, replace=False)]
    t0 = time.time()

    from src.baseline.graph_construction import build_similarity_graphs
    graph = build_similarity_graphs(df, "large")["knn_3"]
    ei, ew = graph_to_edge_arrays(graph, len(X))
    empty_ei, empty_w = np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

    group_rows, node_rows, agree_rows = [], [], []

    def lime_for(label, kind, model, ym, ys, edge_index, edge_w, hops, shap_own=None):
        """LIME per node on its own features; returns group shares of |weights| and per-node overlap with the SHAP own-row part."""
        W = np.zeros((len(nodes), len(names)))
        locals_ = []
        for i, node in enumerate(nodes):
            fn = graph_predict_fn(model, Xs, edge_index, edge_w, node, hops, ym, ys)
            w, score = lime_one(fn, lime_train, Xs[node], names, cat_idx, num_samples=args.lime_samples, seed=SEED)
            for f, v in w.items():
                W[i, f] = v
            locals_.append(score)
        for i, node in enumerate(nodes):
            row = {"model": label, "node": int(node), "lime_local_r2": locals_[i], **base}
            if shap_own is not None:
                top_l, top_s = set(np.argsort(-np.abs(W[i]))[:5]), set(np.argsort(-np.abs(shap_own[i]))[:5])
                row["top5_overlap"] = len(top_l & top_s) / 5
                row["lime_top_is_pore"] = bool(groups[int(np.argmax(np.abs(W[i])))] == "pore geometry")
                row["shap_top_is_pore"] = bool(groups[int(np.argmax(np.abs(shap_own[i])))] == "pore geometry")
            agree_rows.append(row)
        return group_shares(W, list(groups)), float(np.mean(locals_))

    # no-edge GCN: LIME only (its SHAP is in Step 6b)
    m0, ym, ys = train_gcn(Xs, y, empty_ei, empty_w, tr, va, SEED, kind="gcn")
    r2_0 = r2(y[te], gcn_predict(m0, Xs, empty_ei, empty_w, ym, ys)[te])
    sh, loc = lime_for("GCN, no edges", "gcn", m0, ym, ys, empty_ei, empty_w, 1)
    for g, v in sh.items():
        group_rows.append({"model": "GCN, no edges", "method": "LIME (own features)", "group": g, "share": v, "test_r2": r2_0,
                           "n_explained": len(nodes), "note": f"mean local R2 {loc:.2f}", **base})
    logger.info(f"no-edge GCN (test R2 {r2_0:.3f}) LIME: " + ", ".join(f"{g} {v:.1%}" for g, v in sh.items()))

    labels = {"gcn": "GCN, CS2 graph k=3", "graphsage": "GraphSAGE, CS2 graph k=3", "residual_gcn": "Residual GCN, CS2 graph k=3"}
    for kind in MODELS:
        model, ym, ys = train_gcn(Xs, y, ei, ew, tr, va, SEED, kind=kind)
        m_r2 = r2(y[te], gcn_predict(model, Xs, ei, ew, ym, ys)[te])
        gate = float(model.gate.item()) if kind == "residual_gcn" else None
        res = graph_gnn_shap(model, Xs, ei, ew, nodes, HOPS[kind], tr, ys, nsamples=args.shap_samples, seed=SEED)
        own = np.stack([r["own"] for r in res])
        sh_g = group_shares(own, list(groups))
        gap = float(np.mean([r["additivity_gap"] for r in res]))
        share_nb = float(np.mean([r["share_on_neighbours"] for r in res]))
        for g, v in sh_g.items():
            group_rows.append({"model": labels[kind], "method": "SHAP (own features)", "group": g, "share": v, "test_r2": m_r2,
                               "n_explained": len(nodes),
                               "note": f"{share_nb:.0%} of the attribution sits on neighbours' features; additivity gap {gap:.2f} A (approximate)"
                                       + (f"; learned gate {gate:.3f}" if gate is not None else ""), **base})
        for r in res:
            node_rows.append({"model": labels[kind], "node": r["node"], "n_neighbourhood_nodes": r["n_sub"], "share_on_neighbours": r["share_on_neighbours"],
                              "additivity_gap": r["additivity_gap"], "true_pld": float(y[r["node"]]), **base})
        lime_sh, loc = lime_for(labels[kind], kind, model, ym, ys, ei, ew, HOPS[kind], shap_own=own)
        for g, v in lime_sh.items():
            group_rows.append({"model": labels[kind], "method": "LIME (own features)", "group": g, "share": v, "test_r2": m_r2,
                               "n_explained": len(nodes), "note": f"mean local R2 {loc:.2f}", **base})
        logger.info(f"{labels[kind]} (R2 {m_r2:.3f}) SHAP own: " + ", ".join(f"{g} {v:.1%}" for g, v in sh_g.items())
                    + f"; {share_nb:.0%} on neighbours; LIME: " + ", ".join(f"{g} {v:.1%}" for g, v in lime_sh.items()))

    res_dir = config.results_dir
    pd.DataFrame(group_rows).to_csv(os.path.join(res_dir, "step6c_gnn_graph_explainers_groups.csv"), index=False)
    pd.DataFrame(node_rows).to_csv(os.path.join(res_dir, "step6c_gnn_graph_shap_nodes.csv"), index=False)
    ag = pd.DataFrame(agree_rows)
    ag.to_csv(os.path.join(res_dir, "step6c_gnn_lime_vs_shap.csv"), index=False)
    if "top5_overlap" in ag:
        print(ag.dropna(subset=["top5_overlap"]).groupby("model").agg(top5_overlap=("top5_overlap", "mean"), lime_local_r2=("lime_local_r2", "mean"),
                                                                       lime_top_is_pore=("lime_top_is_pore", "mean"), shap_top_is_pore=("shap_top_is_pore", "mean")).round(3))
    make_figure(config, pd.DataFrame(group_rows), pd.DataFrame(node_rows))
    logger.info(f"Step 6c finished in {(time.time() - t0) / 60:.1f} min")


def make_figure(config, groups, nodes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"pore geometry": "#2f6fb5", "linker fingerprint": "#e0a030", "metal": "#8a8a8a"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    g = groups.assign(label=groups["model"] + "\n" + groups["method"].str.split(" ").str[0])
    pivot = g.pivot_table(index="label", columns="group", values="share", aggfunc="first").fillna(0)
    left = np.zeros(len(pivot))
    for c in [c for c in colors if c in pivot.columns]:
        ax.barh(pivot.index, pivot[c], left=left, color=colors[c], label=c)
        left += pivot[c].values
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the explanation (the MOF's own features)")
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax2 = axes[1]
    nb = nodes.groupby("model")["share_on_neighbours"].mean()
    ax2.barh(nb.index, nb.values, color="#c0504d")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("share of SHAP attribution on neighbours")
    ax2.set_yticklabels([])
    fig.tight_layout()
    os.makedirs(config.figures_dir, exist_ok=True)
    os.makedirs(UI_DIR, exist_ok=True)
    fig.savefig(os.path.join(config.figures_dir, "step6c_gnn_graph_explainers.png"), dpi=150)
    fig.savefig(os.path.join(UI_DIR, "gnn_graph_explainers.png"), dpi=110)


if __name__ == "__main__":
    main()
