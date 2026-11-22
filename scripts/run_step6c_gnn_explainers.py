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
from src.explainability.model_explainers import (HOPS, gcn_predict, gnn_shap_values, graph_gnn_shap, graph_predict_fn, group_shapley_exact,
                                                 group_shares, lime_one, standardise_inputs, train_gcn)
from src.explainability.shap_analysis import feature_group, fingerprint_feature_names

logger = get_logger(__name__)
SCRIPT = "scripts/run_step6c_gnn_explainers.py"
UI_DIR = os.path.join("frontend", "public", "step6")
SEED = 0
MODELS = ("gcn", "graphsage", "residual_gcn")


def r2(y, p):
    from sklearn.metrics import r2_score
    return float(r2_score(y, p))


def semantic_knn_graph(E, k=3, block=1500):
    """Undirected top-k cosine-similarity graph from language-model embeddings (same idea as the notebook's semantic graph)."""
    import networkx as nx
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    g = nx.Graph()
    g.add_nodes_from(range(len(E)))
    for start in range(0, len(E), block):
        sim = En[start:start + block] @ En.T
        sim[np.arange(sim.shape[0]), np.arange(sim.shape[0]) + start] = -1.0
        top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        for r in range(sim.shape[0]):
            for c in top[r]:
                g.add_edge(start + r, int(c), weight=float(sim[r, c]))
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-nodes", type=int, default=30)
    ap.add_argument("--lime-samples", type=int, default=5000)
    ap.add_argument("--shap-samples", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--graphs", nargs="+", default=["cs2", "semantic", "semantic_molt5", "cs2_blackhole"], choices=["cs2", "semantic", "semantic_molt5", "cs2_blackhole"])
    args = ap.parse_args()

    config = RuntimeConfig(language_model="chemberta")
    df = load_dataset("large")
    y = df["pld_value"].values.astype(np.float32)
    X, meta = build_features(df, feature_scheme="fingerprint")
    names = fingerprint_feature_names(meta)
    groups = np.array([feature_group(n) for n in names])
    test_nodes = compute_global_fixed_test_nodes(len(df), config)
    te = np.array(sorted(test_nodes))
    non_test = np.array([i for i in range(len(X)) if i not in test_nodes])
    Xs, *_ = standardise_inputs(X, non_test)
    cat_idx = list(np.where(((X == 0) | (X == 1)).all(axis=0))[0])
    data_sha = hashlib.sha256(open(config.dataset_path, "rb").read()).hexdigest()[:16]
    base = {"script": SCRIPT, "data_sha256_16": data_sha, "split_random_state": config.split_random_state}
    nodes = np.sort(np.random.default_rng(0).choice(te, size=min(args.n_nodes, len(te)), replace=False))  # same nodes for every model/seed
    t0 = time.time()

    from src.baseline.graph_construction import build_similarity_graphs
    graphs = {}
    if "cs2" in args.graphs:
        graphs["CS2 graph k=3"] = build_similarity_graphs(df, "large")["knn_3"]
    if "semantic" in args.graphs:
        from src.language.embeddings import cached_embed
        graphs["semantic graph (ChemBERTa) k=3"] = semantic_knn_graph(cached_embed(df["linker_smiles"].tolist(), config).astype(np.float32), 3)
    if "semantic_molt5" in args.graphs:
        from src.language.embeddings import cached_embed
        molt5 = RuntimeConfig(language_model="molt5")
        graphs["semantic graph (MolT5) k=3"] = semantic_knn_graph(cached_embed(df["linker_smiles"].tolist(), molt5).astype(np.float32), 3)
    if "cs2_blackhole" in args.graphs:
        # Black Hole pruning to 30% retention of the CS2 k=3 graph, exactly as in the Step 5 sweep (ported code + test nodes restored)
        from src.baseline.black_hole_sparsification import apply_black_hole_sparsification
        from src.evaluation.run_comparison import restore_test_nodes
        cs2 = graphs.get("CS2 graph k=3") or build_similarity_graphs(df, "large")["knn_3"]
        weights = (config.gravity_degree_weight, config.gravity_betweenness_weight, config.gravity_edge_weight_sum_weight)
        pruned = apply_black_hole_sparsification(cs2, df, weights, threshold=1 - 0.3)["graph"]
        graphs["CS2 graph k=3, Black Hole 30%"] = restore_test_nodes(pruned, cs2, set(test_nodes))
        logger.info(f"Black Hole 30%: {pruned.number_of_nodes()} of {len(df)} nodes kept, {graphs['CS2 graph k=3, Black Hole 30%'].number_of_edges()} edges")
    empty_ei, empty_w = np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    group_rows, node_rows, agree_rows, check_rows = [], [], [], []
    labels = {"gcn": "GCN", "graphsage": "GraphSAGE", "residual_gcn": "Residual GCN"}

    for seed in args.seeds:
        tr, va, _ = make_splits(len(X), test_nodes, seed=seed)
        rng = np.random.default_rng(seed + 100)
        lime_train = Xs[rng.choice(tr, size=2000, replace=False)]
        info = {**base, "seed": seed}

        def lime_for(label, graph_name, model, ym, ys, edge_index, edge_w, hops, shap_own=None):
            W, scores = np.zeros((len(nodes), len(names))), []
            for i, node in enumerate(nodes):
                fn = graph_predict_fn(model, Xs, edge_index, edge_w, node, hops, ym, ys)
                w, score = lime_one(fn, lime_train, Xs[node], names, cat_idx, num_samples=args.lime_samples, seed=seed)
                for f, v in w.items():
                    W[i, f] = v
                scores.append(score)
            for i, node in enumerate(nodes):
                row = {"model": label, "graph": graph_name, "node": int(node), "lime_local_r2": scores[i], **info}
                if shap_own is not None:
                    row["top5_overlap"] = len(set(np.argsort(-np.abs(W[i]))[:5]) & set(np.argsort(-np.abs(shap_own[i]))[:5])) / 5
                    row["lime_top_is_pore"] = bool(groups[int(np.argmax(np.abs(W[i])))] == "pore geometry")
                    row["shap_top_is_pore"] = bool(groups[int(np.argmax(np.abs(shap_own[i])))] == "pore geometry")
                agree_rows.append(row)
            return (group_shares(W, list(groups)), group_shares(W, list(groups), net=True)), float(np.mean(scores))

        # no-edge GCN: LIME + the exact group-Shapley check of the expected-gradients SHAP
        m0, ym, ys = train_gcn(Xs, y, empty_ei, empty_w, tr, va, seed, kind="gcn")
        r2_0 = r2(y[te], gcn_predict(m0, Xs, empty_ei, empty_w, ym, ys)[te])
        (sh, sh_net), loc = lime_for("GCN, no edges", "none", m0, ym, ys, empty_ei, empty_w, 1)
        for g, v in sh.items():
            group_rows.append({"model": "GCN, no edges", "graph": "none", "method": "LIME (own features)", "group": g, "share": v, "share_net": sh_net[g], "test_r2": r2_0,
                               "n_explained": len(nodes), "note": f"mean local R2 {loc:.2f}", **info})
        group_idx = [np.where(groups == g)[0] for g in ("pore geometry", "linker fingerprint", "metal")]
        ex_idx = np.sort(np.random.default_rng(seed + 200).choice(te, size=150, replace=False))
        bg_idx = np.sort(np.random.default_rng(seed + 300).choice(tr, size=100, replace=False))
        predict_rows = lambda rows: gcn_predict(m0, rows.astype(np.float32), empty_ei, empty_w, ym, ys)
        exact = group_shapley_exact(predict_rows, Xs[ex_idx], Xs[bg_idx], group_idx)
        eg = gnn_shap_values(m0, Xs, bg_idx, ex_idx, ys, seed=seed)["shap_values"]
        eg_g = np.stack([eg[:, gi].sum(axis=1) for gi in group_idx], axis=1)
        for k_, gname in enumerate(("pore geometry", "linker fingerprint", "metal")):
            check_rows.append({"group": gname, "mean_abs_exact_shapley": float(np.abs(exact[:, k_]).mean()),
                               "mean_abs_expected_gradients": float(np.abs(eg_g[:, k_]).mean()),
                               "pearson_corr": float(np.corrcoef(exact[:, k_], eg_g[:, k_])[0, 1]),
                               "mean_abs_difference_A": float(np.abs(exact[:, k_] - eg_g[:, k_]).mean()), "n_rows": len(ex_idx), **info})
        logger.info(f"seed {seed}: no-edge GCN R2 {r2_0:.3f}; EG vs exact group Shapley: " + ", ".join(
            f"{c['group']} r={c['pearson_corr']:.2f}" for c in check_rows[-3:]))

        for graph_name, graph in graphs.items():
            ei, ew = graph_to_edge_arrays(graph, len(X))
            for kind in MODELS:
                model, ym, ys = train_gcn(Xs, y, ei, ew, tr, va, seed, kind=kind)
                m_r2 = r2(y[te], gcn_predict(model, Xs, ei, ew, ym, ys)[te])
                gate = float(model.gate.item()) if kind == "residual_gcn" else None
                label = labels[kind]
                res = graph_gnn_shap(model, Xs, ei, ew, nodes, HOPS[kind], tr, ys, nsamples=args.shap_samples, seed=seed)
                own = np.stack([r["own"] for r in res])
                sh_g, sh_g_net = group_shares(own, list(groups)), group_shares(own, list(groups), net=True)
                gap, share_nb = float(np.mean([r["additivity_gap"] for r in res])), float(np.mean([r["share_on_neighbours"] for r in res]))
                for g, v in sh_g.items():
                    group_rows.append({"model": label, "graph": graph_name, "method": "SHAP (own features)", "group": g, "share": v, "share_net": sh_g_net[g], "test_r2": m_r2,
                                       "n_explained": len(nodes), "note": f"{share_nb:.0%} on neighbours; additivity gap {gap:.2f} A"
                                       + (f"; gate {gate:.3f}" if gate is not None else ""), **info})
                for r in res:
                    node_rows.append({"model": label, "graph": graph_name, "node": r["node"], "n_neighbourhood_nodes": r["n_sub"],
                                      "share_on_neighbours": r["share_on_neighbours"], "additivity_gap": r["additivity_gap"],
                                      "true_pld": float(y[r["node"]]), "test_r2_of_model": m_r2, **info})
                (lime_sh, lime_net), loc = lime_for(label, graph_name, model, ym, ys, ei, ew, HOPS[kind], shap_own=own)
                for g, v in lime_sh.items():
                    group_rows.append({"model": label, "graph": graph_name, "method": "LIME (own features)", "group": g, "share": v, "share_net": lime_net[g], "test_r2": m_r2,
                                       "n_explained": len(nodes), "note": f"mean local R2 {loc:.2f}", **info})
                logger.info(f"seed {seed} | {label} on {graph_name} (R2 {m_r2:.3f}): SHAP own " + ", ".join(f"{g} {v:.0%}" for g, v in sh_g.items())
                            + f"; {share_nb:.0%} on neighbours")

    res_dir = config.results_dir
    groups_df, nodes_df, agree_df, check_df = map(pd.DataFrame, (group_rows, node_rows, agree_rows, check_rows))
    groups_df.to_csv(os.path.join(res_dir, "step6c_gnn_graph_explainers_groups.csv"), index=False)
    nodes_df.to_csv(os.path.join(res_dir, "step6c_gnn_graph_shap_nodes.csv"), index=False)
    agree_df.to_csv(os.path.join(res_dir, "step6c_gnn_lime_vs_shap.csv"), index=False)
    check_df.to_csv(os.path.join(res_dir, "step6c_expected_gradients_check.csv"), index=False)
    pd.set_option("display.width", 220)
    print(check_df.groupby("group")[["mean_abs_exact_shapley", "mean_abs_expected_gradients", "pearson_corr", "mean_abs_difference_A"]].agg(["mean", "std"]).round(3))
    print(nodes_df.groupby(["graph", "model"]).agg(r2=("test_r2_of_model", "mean"), on_neighbours=("share_on_neighbours", "mean")).round(3))
    make_figure(config, groups_df, nodes_df)
    logger.info(f"Step 6c finished in {(time.time() - t0) / 60:.1f} min")


def make_figure(config, groups, nodes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"pore geometry": "#2f6fb5", "linker fingerprint": "#e0a030", "metal": "#8a8a8a"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), gridspec_kw={"width_ratios": [2.2, 1]})
    g = groups.assign(label=groups["model"] + "\n" + groups["graph"].str.replace(" graph", "") + " | " + groups["method"].str.split(" ").str[0])
    pivot = g.groupby(["label", "group"])["share_net"].mean().unstack().fillna(0)
    pivot = pivot.loc[sorted(pivot.index, key=lambda t: (t.split(" | ")[1], t))]
    left = np.zeros(len(pivot))
    for c in [c for c in colors if c in pivot.columns]:
        axes[0].barh(pivot.index, pivot[c], left=left, color=colors[c], label=c)
        left += pivot[c].values
    axes[0].set_xlim(0, 1)
    axes[0].tick_params(axis="y", labelsize=7)
    axes[0].set_xlabel("net share of the explanation (the MOF's own features; mean of 3 seeds)")
    axes[0].legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False)
    per_seed = nodes.groupby(["graph", "model", "seed"])["share_on_neighbours"].mean().groupby(["graph", "model"]).agg(["mean", "std"]).fillna(0)
    labels_ = [f"{m}\n{gr.replace(' graph', '')}" for gr, m in per_seed.index]
    axes[1].barh(labels_, per_seed["mean"], xerr=per_seed["std"], color="#c0504d")
    axes[1].set_xlim(0, 1)
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].set_xlabel("share of SHAP attribution on neighbours")
    fig.tight_layout()
    os.makedirs(config.figures_dir, exist_ok=True)
    os.makedirs(UI_DIR, exist_ok=True)
    fig.savefig(os.path.join(config.figures_dir, "step6c_gnn_graph_explainers.png"), dpi=150)
    fig.savefig(os.path.join(UI_DIR, "gnn_graph_explainers.png"), dpi=110)


if __name__ == "__main__":
    main()
