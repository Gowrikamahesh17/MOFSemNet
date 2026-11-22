#!/usr/bin/env python
"""Step 6b - more explainers: SHAP for XGBoost, SHAP for the (no-edge) GCN, GNNExplainer for the GCN with a graph, and LIME.

Same data, features, random test set (split state 42) and seed-0 split as Step 6. Writes results/step6b_*.csv, two figures
(outputs/figures and frontend/public/step6/) used by the web console.

Usage:
    PYTHONPATH=. python scripts/run_step6b_explainers.py [--parts xgb,gnn_shap,gnnexplainer,lime] [--n-shap 300] [--n-lime 100] [--n-gnnexp 40]
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
from src.explainability.model_explainers import (gcn_predict, gnn_explainer_summary, gnn_shap_values, group_shares,
                                                 lime_explanations, standardise_inputs, train_gcn)
from src.explainability.shap_analysis import compute_shap_values, feature_group, fingerprint_feature_names

logger = get_logger(__name__)
SCRIPT = "scripts/run_step6b_explainers.py"
UI_DIR = os.path.join("frontend", "public", "step6")
SEED = 0


def r2(y, p):
    from sklearn.metrics import r2_score
    return float(r2_score(y, p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts", default="xgb,gnn_shap,gnnexplainer,lime")
    ap.add_argument("--n-shap", type=int, default=300)
    ap.add_argument("--n-lime", type=int, default=100)
    ap.add_argument("--n-gnnexp", type=int, default=40)
    ap.add_argument("--lime-samples", type=int, default=15000)
    args = ap.parse_args()
    parts = {p.strip() for p in args.parts.split(",")}

    config = RuntimeConfig(language_model="chemberta")
    os.makedirs(config.results_dir, exist_ok=True)
    df = load_dataset("large")
    y = df["pld_value"].values.astype(np.float32)
    X, meta = build_features(df, feature_scheme="fingerprint")
    names = fingerprint_feature_names(meta)
    groups = [feature_group(n) for n in names]
    test_nodes = compute_global_fixed_test_nodes(len(df), config)
    tr, va, te = make_splits(len(X), test_nodes, seed=SEED)
    te = np.array(sorted(test_nodes))
    data_sha = hashlib.sha256(open(config.dataset_path, "rb").read()).hexdigest()[:16]
    base = {"script": SCRIPT, "data_sha256_16": data_sha, "split_random_state": config.split_random_state, "seed": SEED, "n_test": len(te)}
    rng = np.random.default_rng(SEED)
    rows = []  # one row per (method, group)

    def add_groups(method, model_name, r2_test, shares, n_explained, note="", net=None):
        for g, s in shares.items():
            rows.append({"method": method, "model": model_name, "group": g, "share": s, "share_net": (net or {}).get(g), "test_r2": r2_test,
                         "n_explained": n_explained, "note": note, **base})
        logger.info(f"{method}: " + ", ".join(f"{g} {s:.1%}" for g, s in shares.items()))

    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, min_samples_leaf=5).fit(X[tr], y[tr])
    rf_r2 = r2(y[te], rf.predict(X[te]))
    t0 = time.time()

    # ---------------- XGBoost: TreeSHAP ----------------
    xgbm = None
    if parts & {"xgb", "lime"}:
        import xgboost as xgb
        xgbm = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.5,
                                tree_method="hist", early_stopping_rounds=50, random_state=SEED, n_jobs=-1, verbosity=0,
                                base_score=float(y[tr].mean()))  # explicit scalar: shap 0.48 cannot parse xgboost 3.x's automatic "[x]" base score
        xgbm.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        xgb_r2 = r2(y[te], xgbm.predict(X[te]))
    if "xgb" in parts:
        out = compute_shap_values(xgbm, X[te], names, max_samples=500, seed=SEED)
        add_groups("SHAP (TreeSHAP)", "XGBoost", xgb_r2, {g: d["share"] for g, d in out["groups"].items()}, len(out["row_index"]),
                   f"additivity error {out['max_additivity_error']:.4f}", net={g: d["net_share"] for g, d in out["groups"].items()})
        pd.DataFrame([{"feature": f, "mean_abs_shap": v, "test_r2": xgb_r2, **base}
                      for f, v in sorted(out["mean_abs_shap"].items(), key=lambda kv: -kv[1])]).to_csv(
            os.path.join(config.results_dir, "step6b_xgboost_shap_features.csv"), index=False)
        rf_out = compute_shap_values(rf, X[te], names, max_samples=500, seed=SEED)
        add_groups("SHAP (TreeSHAP)", "Random Forest", rf_r2, {g: d["share"] for g, d in rf_out["groups"].items()}, len(rf_out["row_index"]),
                   "500 MOFs, same setting as Step 6a", net={g: d["net_share"] for g, d in rf_out["groups"].items()})

    # ---------------- GNNs ----------------
    if parts & {"gnn_shap", "gnnexplainer"}:
        non_test = np.array([i for i in range(len(X)) if i not in test_nodes])
        Xs, _mu, _sd, _nb = standardise_inputs(X, non_test)
        empty_ei, empty_w = np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
        if "gnn_shap" in parts:
            model, ym, ys = train_gcn(Xs, y, empty_ei, empty_w, tr, va, SEED)
            gcn_r2 = r2(y[te], gcn_predict(model, Xs, empty_ei, empty_w, ym, ys)[te])
            ex_idx = np.sort(rng.choice(te, size=min(args.n_shap, len(te)), replace=False))
            bg_idx = np.sort(rng.choice(tr, size=100, replace=False))
            out = gnn_shap_values(model, Xs, bg_idx, ex_idx, ys, seed=SEED)
            add_groups("SHAP (expected gradients)", "GCN, no edges (training fix)", gcn_r2, group_shares(out["shap_values"], groups), len(ex_idx),
                       f"additivity gap mean {out['additivity_error_mean']:.3f} A, max {out['additivity_error_max']:.3f} A (approximate method)",
                       net=group_shares(out["shap_values"], groups, net=True))
            mean_abs = np.abs(out["shap_values"]).mean(0)
            pd.DataFrame([{"feature": n, "mean_abs_shap": float(v), "test_r2": gcn_r2, **base} for n, v in sorted(zip(names, mean_abs), key=lambda t: -t[1])]).to_csv(
                os.path.join(config.results_dir, "step6b_gcn_shap_features.csv"), index=False)
        if "gnnexplainer" in parts:
            from src.baseline.graph_construction import build_similarity_graphs
            graph = build_similarity_graphs(df, "large")["knn_3"]
            ei, ew = graph_to_edge_arrays(graph, len(X))
            gmodel, ym, ys = train_gcn(Xs, y, ei, ew, tr, va, SEED)
            g_r2 = r2(y[te], gcn_predict(gmodel, Xs, ei, ew, ym, ys)[te])
            nodes = np.sort(rng.choice(te, size=min(args.n_gnnexp, len(te)), replace=False))
            node_rows, own_masks = gnn_explainer_summary(gmodel, Xs, ei, ew, nodes)
            pd.DataFrame([{**r, "true_pld": float(y[r["node"]]), "test_r2_of_model": g_r2, **base} for r in node_rows]).to_csv(
                os.path.join(config.results_dir, "step6b_gnnexplainer_nodes.csv"), index=False)
            add_groups("GNNExplainer (feature mask, own row)", "GCN, CS2 graph k=3 (training fix)", g_r2, group_shares(own_masks, groups), len(nodes),
                       f"share of mask on the node's own features {np.mean([r['share_on_own_features'] for r in node_rows]):.1%}; "
                       f"mean neighbourhood {np.mean([r['n_neighbourhood_nodes'] for r in node_rows]):.1f} nodes")

    # ---------------- LIME ----------------
    if "lime" in parts:
        lime_rows = np.sort(rng.choice(te, size=min(args.n_lime, len(te)), replace=False))
        is_bin = ((X == 0) | (X == 1)).all(axis=0)
        cat_idx = list(np.where(is_bin)[0])
        train_sample = X[rng.choice(tr, size=2000, replace=False)]
        agree_rows, feat_tables = [], {}
        for label, model, m_r2 in (("Random Forest", rf, rf_r2), ("XGBoost", xgbm, xgb_r2)):
            weights, scores = lime_explanations(model.predict, train_sample, X[lime_rows], names, cat_idx, num_features=15,
                                                num_samples=args.lime_samples, seed=SEED)
            W = np.zeros((len(lime_rows), len(names)))
            for i, w in enumerate(weights):
                for f, v in w.items():
                    W[i, f] = v
            shap_out = compute_shap_values(model, X[lime_rows], names, max_samples=len(lime_rows), seed=SEED)
            S = shap_out["shap_values"]
            add_groups("LIME (15 features per MOF)", label, m_r2, group_shares(W, groups), len(lime_rows),
                       f"mean local-model R2 {np.mean(scores):.2f}", net=group_shares(W, groups, net=True))
            for i in range(len(lime_rows)):
                top_l = set(np.argsort(-np.abs(W[i]))[:5])
                top_s = set(np.argsort(-np.abs(S[i]))[:5])
                agree_rows.append({"model": label, "node": int(lime_rows[i]), "top5_overlap": len(top_l & top_s) / 5,
                                   "lime_top_feature": names[int(np.argmax(np.abs(W[i])))], "shap_top_feature": names[int(np.argmax(np.abs(S[i])))],
                                   "lime_r2_local": scores[i], "true_pld": float(y[lime_rows[i]]), **base})
            feat_tables[label] = (W, S)
            pd.DataFrame([{"feature": n, "mean_abs_lime_weight": float(v), "mean_abs_shap_same_mofs": float(s), **base}
                          for n, v, s in sorted(zip(names, np.abs(W).mean(0), np.abs(S).mean(0)), key=lambda t: -t[1])]).to_csv(
                os.path.join(config.results_dir, f"step6b_lime_features_{label.lower().replace(' ', '_')}.csv"), index=False)
        pd.DataFrame(agree_rows).to_csv(os.path.join(config.results_dir, "step6b_lime_vs_shap_agreement.csv"), index=False)
        summ = pd.DataFrame(agree_rows).groupby("model").agg(top5_overlap=("top5_overlap", "mean"),
                                                             lime_top_is_pore=("lime_top_feature", lambda s: s.str.startswith("pore_").mean()),
                                                             shap_top_is_pore=("shap_top_feature", lambda s: s.str.startswith("pore_").mean()))
        print(summ.round(3))
        make_lime_example(config, names, lime_rows, feat_tables["Random Forest"], y)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(config.results_dir, "step6b_method_groups.csv"), index=False)
    print(res.drop(columns=["script", "data_sha256_16", "split_random_state", "seed", "n_test"]).round(3).to_string())
    make_comparison_figure(config, res)
    logger.info(f"Step 6b finished in {(time.time() - t0) / 60:.1f} min")


def _save(fig, config, name):
    os.makedirs(config.figures_dir, exist_ok=True)
    os.makedirs(UI_DIR, exist_ok=True)
    fig.savefig(os.path.join(config.figures_dir, name), dpi=150)
    fig.savefig(os.path.join(UI_DIR, name.replace("step6b_", "")), dpi=110)


def make_comparison_figure(config, res):
    """Two panels: group shares as the sum of per-column mean |value| (left) and as each group's NET contribution per MOF (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = res.assign(label=res["method"].str.split(" \\(").str[0] + "\n" + res["model"])
    colors = {"pore geometry": "#2f6fb5", "linker fingerprint": "#e0a030", "metal": "#8a8a8a"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, col, title in ((axes[0], "share", "sum of per-column mean |value|"), (axes[1], "share_net", "net contribution of each group per MOF")):
        pivot = res.pivot_table(index="label", columns="group", values=col, aggfunc="first")
        pivot = pivot.reindex(res["label"].drop_duplicates())
        left = np.zeros(len(pivot))
        for c in [c for c in colors if c in pivot.columns]:
            vals = pivot[c].fillna(0).values
            ax.barh(pivot.index, vals, left=left, color=colors[c], label=c)
            left += vals
        ax.set_xlim(0, 1)
        ax.set_xlabel(title)
        ax.invert_yaxis()
    axes[0].tick_params(axis="y", labelsize=8)
    axes[0].legend(loc="lower center", ncol=3, bbox_to_anchor=(1.0, 1.0), frameon=False)
    fig.tight_layout()
    _save(fig, config, "step6b_method_comparison.png")


def make_lime_example(config, names, lime_rows, tables, y):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W, S = tables
    cand = [i for i in range(len(lime_rows)) if 6.0 <= y[lime_rows[i]] <= 10.0] or list(range(len(lime_rows)))
    i = cand[0]
    top = list(dict.fromkeys(list(np.argsort(-np.abs(W[i]))[:6]) + list(np.argsort(-np.abs(S[i]))[:6])))[:8][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    h = 0.38
    ax.barh(np.arange(len(top)) + h / 2, [S[i][j] for j in top], height=h, label="SHAP", color="#2f6fb5")
    ax.barh(np.arange(len(top)) - h / 2, [W[i][j] for j in top], height=h, label="LIME", color="#e0a030")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([names[j] for j in top])
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("contribution to the predicted PLD (A)")
    ax.set_title(f"One MOF (node {int(lime_rows[i])}, true PLD {y[lime_rows[i]]:.1f} A): Random Forest explanation")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, config, "step6b_lime_example.png")


if __name__ == "__main__":
    main()
