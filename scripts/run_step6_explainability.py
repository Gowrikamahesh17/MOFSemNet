#!/usr/bin/env python
"""Step 6 - trustworthiness analysis of the final model (Random Forest on fingerprint features).

  6a  SHAP: which input columns drive the predictions (fingerprint bits / pore geometry / metal / embedding)
  6b  Embedding ablation: do language-model embeddings carry usable chemistry? Rebuild the semantic graph
      (and the Random Forest inputs) from shuffled / random / truncated embeddings and compare R2
  6c  Calibrated confidence: split conformal prediction intervals, with measured coverage on the test set

Same fixed test set (config.split_random_state=42, 15% random MOFs) and seeds 0-4 as Steps 5 and 5b. The
conformal calibration set is Step 5's validation split, disjoint from training and test.

Usage:
    PYTHONPATH=. python scripts/run_step6_explainability.py --model chemberta [--parts shap,columns,confidence,ablation]
"""

import argparse
import hashlib
import os
import time

import numpy as np
import pandas as pd

from src.baseline.data_ingestion import load_dataset
from src.baseline.feature_engineering import build_features
from src.baseline.gnn_training import make_splits
from src.common.config import LANGUAGE_MODEL_CHOICES, RuntimeConfig
from src.common.logging_setup import get_logger
from src.evaluation.run_comparison import compute_global_fixed_test_nodes
from src.explainability.confidence_calibration import calibrate_confidence, evaluate_coverage
from src.explainability.embedding_ablation import ablate_embeddings, run_embedding_ablation
from src.explainability.shap_analysis import compute_shap_values, fingerprint_feature_names
from src.language.embeddings import cached_embed

logger = get_logger(__name__)
SCRIPT = "scripts/run_step6_explainability.py"
PLD_BINS = [(-np.inf, 2.4, "nonporous (<2.4)"), (2.4, 4.0, "small (2.4-4)"), (4.0, 8.0, "medium (4-8)"), (8.0, np.inf, "large (>=8)")]


def forest(seed, min_samples_leaf=1):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1, min_samples_leaf=min_samples_leaf)


def r2(y, p):
    from sklearn.metrics import r2_score
    return float(r2_score(y, p))


def cs2_hub_test_nodes(df, config):
    """Case Study 2's test set: about 4% highest-gravity nodes per Louvain community of the Case Study 2 graph
    (0.9 x Tanimoto + 0.1 x same-metal, kNN k=3), as selected by the ported Black Hole code. Saved for traceability."""
    from networkx.algorithms.community import louvain_communities
    from src.baseline.black_hole_sparsification import calculate_gravity_per_community
    from src.baseline.graph_construction import build_similarity_graphs
    graph = build_similarity_graphs(df, "large")["knn_3"]
    weights = (config.gravity_degree_weight, config.gravity_betweenness_weight, config.gravity_edge_weight_sum_weight)
    communities = louvain_communities(graph, seed=42)
    nodes = calculate_gravity_per_community(graph, communities, weights)[4]
    pd.DataFrame({"node_index": sorted(nodes), "refcode": df["refcode"].values[sorted(nodes)]}).to_csv(
        os.path.join(config.results_dir, "cs2_hub_test_nodes.csv"), index=False)
    return set(nodes)


def meta_cols(args, config, data_sha, test_nodes):
    return {"test_set": args.test_set, "language_model": args.model, "script": SCRIPT, "data_sha256_16": data_sha,
            "split_random_state": config.split_random_state, "n_test": len(test_nodes)}


def run_shap(args, config, X, E, meta, labels, test_nodes, base):
    from src.explainability.shap_analysis import fingerprint_feature_names  # noqa: F811
    tr, _va, _te = make_splits(len(X), test_nodes, seed=0)
    feats = {"fingerprint": (X, fingerprint_feature_names(meta)),
             "fingerprint+embedding": (np.hstack([X, E]).astype(np.float32), fingerprint_feature_names(meta, E.shape[1]))}
    feat_rows, group_rows = [], []
    for name, (M, names) in feats.items():
        # TreeSHAP is numerically unreliable on fully grown trees, so the explained forest needs min_samples_leaf > 1
        model = forest(0, min_samples_leaf=args.shap_min_leaf).fit(M[tr], labels[tr])
        te = np.array(sorted(test_nodes))
        out = compute_shap_values(model, M[te], names, max_samples=args.shap_samples, seed=0)
        test_r2 = r2(labels[te], model.predict(M[te]))
        for fname, v in sorted(out["mean_abs_shap"].items(), key=lambda kv: -kv[1]):
            feat_rows.append({"features": name, "feature": fname, "mean_abs_shap": v, "n_explained": len(out["row_index"]),
                              "test_r2": test_r2, "min_samples_leaf": args.shap_min_leaf, **base})
        for g, d in out["groups"].items():
            group_rows.append({"features": name, "group": g, "n_features": d["n_features"],
                               "mean_abs_shap_sum": d["mean_abs_shap_sum"], "share": d["share"], "test_r2": test_r2,
                               "min_samples_leaf": args.shap_min_leaf, "max_additivity_error": out["max_additivity_error"], **base})
        logger.info(f"SHAP {name}: test R2 {test_r2:.3f}, additivity error {out['max_additivity_error']:.4f}, groups " + ", ".join(f"{g} {d['share']:.1%}" for g, d in out["groups"].items()))
    feat_df, group_df = pd.DataFrame(feat_rows), pd.DataFrame(group_rows)
    feat_df.to_csv(os.path.join(config.results_dir, f"step6_shap_features_{args.tag}.csv"), index=False)
    group_df.to_csv(os.path.join(config.results_dir, f"step6_shap_groups_{args.tag}.csv"), index=False)
    print(group_df.round(3).to_string())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(config.figures_dir, exist_ok=True)
        top = feat_df[feat_df.features == "fingerprint"].head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top.feature, top.mean_abs_shap, color="#2f6fb5")
        ax.set_xlabel("mean |SHAP| (change in predicted PLD, A)")
        ax.set_title("Random Forest, fingerprint features: top 15 columns")
        fig.tight_layout()
        fig.savefig(os.path.join(config.figures_dir, f"step6_shap_top_features_{args.tag}.png"), dpi=150)
        plt.close(fig)
    except Exception as exc:  # plotting is optional
        logger.warning(f"figure skipped: {exc}")


def run_columns(args, config, X, labels, test_nodes, base):
    """How much each kind of input column explains on its own (Random Forest, 5 seeds)."""
    n_fp = 1024
    sets = {"all columns": list(range(X.shape[1])),
            "pore geometry only (2 columns)": [n_fp, n_fp + 1],
            "Largest Free Sphere only": [n_fp + 1],
            "linker fingerprint + metal (no pore columns)": list(range(n_fp)) + list(range(n_fp + 2, X.shape[1]))}
    te = np.array(sorted(test_nodes))
    rows = []
    for name, cols in sets.items():
        for seed in config.seeds:
            tr, _va, _ = make_splits(len(X), test_nodes, seed=seed)
            p = forest(seed).fit(X[tr][:, cols], labels[tr]).predict(X[te][:, cols])
            rows.append({"columns": name, "n_columns": len(cols), "seed": seed, "r2": r2(labels[te], p),
                         "mae": float(np.mean(np.abs(labels[te] - p))), **base})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(config.results_dir, f"step6_column_groups_{args.tag}_per_seed.csv"), index=False)
    agg = per_seed.groupby(["columns", "n_columns"], sort=False).agg(
        n_seeds=("seed", "nunique"), r2_mean=("r2", "mean"), r2_std=("r2", "std"), mae_mean=("mae", "mean")).reset_index()
    agg.to_csv(os.path.join(config.results_dir, f"step6_column_groups_{args.tag}.csv"), index=False)
    print(agg.round(3).to_string())


def run_confidence(args, config, X, labels, test_nodes, base):
    rows, bin_rows = [], []
    te = np.array(sorted(test_nodes))
    for seed in config.seeds:
        tr, va, _ = make_splits(len(X), test_nodes, seed=seed)
        model = forest(seed).fit(X[tr], labels[tr])
        test_r2 = r2(labels[te], model.predict(X[te]))
        for alpha in args.alphas:
            for mode in ("absolute", "normalized"):
                cal = calibrate_confidence(model, X[va], labels[va], alpha=alpha, mode=mode)
                res = evaluate_coverage(model, X[te], labels[te], cal)
                err = np.abs(labels[te] - model.predict(X[te]))
                from scipy.stats import spearmanr
                rho = float(spearmanr(res["width"], err)[0]) if mode == "normalized" else float("nan")
                rows.append({"seed": seed, "alpha": alpha, "target_coverage": 1 - alpha, "interval": mode,
                             "coverage": res["coverage"], "mean_width": res["mean_width"], "median_width": res["median_width"],
                             "width_error_spearman": rho, "q_hat": cal["q_hat"], "n_calibration": cal["n_calibration"],
                             "test_r2": test_r2, **base})
                y = labels[te]
                for lo, hi, label in PLD_BINS:
                    m = (y >= lo) & (y < hi)
                    if m.sum():
                        bin_rows.append({"seed": seed, "alpha": alpha, "interval": mode, "pld_bin": label, "n": int(m.sum()),
                                         "coverage": float(res["inside"][m].mean()), "mean_width": float(res["width"][m].mean()), **base})
        logger.info(f"confidence seed {seed} done")
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(config.results_dir, f"step6_confidence_{args.tag}_per_seed.csv"), index=False)
    summary = (per_seed.groupby(["alpha", "interval"]).agg(
        n_seeds=("seed", "nunique"), coverage_mean=("coverage", "mean"), coverage_std=("coverage", "std"),
        width_mean=("mean_width", "mean"), median_width_mean=("median_width", "mean"),
        width_error_spearman=("width_error_spearman", "mean"), test_r2=("test_r2", "mean")).reset_index())
    summary.to_csv(os.path.join(config.results_dir, f"step6_confidence_{args.tag}.csv"), index=False)
    bins = (pd.DataFrame(bin_rows).groupby(["alpha", "interval", "pld_bin"]).agg(
        n=("n", "mean"), coverage_mean=("coverage", "mean"), width_mean=("mean_width", "mean")).reset_index())
    bins.to_csv(os.path.join(config.results_dir, f"step6_confidence_by_pld_{args.tag}.csv"), index=False)
    print(summary.round(3).to_string())
    print(bins.round(3).to_string())


def run_ablation(args, config, df, X, E, labels, test_nodes, base):
    per_seed_path = os.path.join(config.results_dir, f"step6_embedding_ablation_{args.tag}_per_seed.csv")
    rows = []

    def save():
        pd.DataFrame(rows).to_csv(per_seed_path, index=False)

    te = np.array(sorted(test_nodes))
    # (i) Random Forest on fingerprint + (ablated) embedding: cheap, run first
    for kind in ("none", "shuffle_rows", "random_replacement", "truncate_dims"):
        for seed in config.seeds:
            Ea = ablate_embeddings(E, kind, seed=seed, keep_dims=args.keep_dims)
            M = np.hstack([X, Ea]).astype(np.float32)
            tr, _va, _ = make_splits(len(X), test_nodes, seed=seed)
            model = forest(seed).fit(M[tr], labels[tr])
            p = model.predict(M[te])
            from sklearn.metrics import mean_absolute_error
            rows.append({"experiment": "rf_input", "ablation": kind, "model": "random_forest", "k": 0, "seed": seed,
                         "mae": float(mean_absolute_error(labels[te], p)), "rmse": float(np.sqrt(np.mean((labels[te] - p) ** 2))),
                         "r2": r2(labels[te], p), "n_edges": 0, "elapsed_seconds": float("nan"), **base})
        save()
        logger.info(f"RF input ablation {kind} done")
    for seed in config.seeds:  # reference without any embedding
        tr, _va, _ = make_splits(len(X), test_nodes, seed=seed)
        p = forest(seed).fit(X[tr], labels[tr]).predict(X[te])
        rows.append({"experiment": "rf_input", "ablation": "no_embedding", "model": "random_forest", "k": 0, "seed": seed,
                     "mae": float(np.mean(np.abs(labels[te] - p))), "rmse": float(np.sqrt(np.mean((labels[te] - p) ** 2))),
                     "r2": r2(labels[te], p), "n_edges": 0, "elapsed_seconds": float("nan"), **base})
    save()
    # (ii) GNNs on the semantic graph built from ablated embeddings
    for kind in ("none", "shuffle_rows", "random_replacement", "truncate_dims"):
        for r in run_embedding_ablation(E, kind, config, X=X, labels=labels, test_nodes=test_nodes, k=args.k,
                                        keep_dims=args.keep_dims):
            rows.append({"experiment": "semantic_graph", **r, **base})
        save()
        logger.info(f"semantic-graph ablation {kind} done")
    per_seed = pd.DataFrame(rows)
    agg = (per_seed.groupby(["experiment", "ablation", "model", "k"], sort=False).agg(
        n_seeds=("seed", "nunique"), mae_mean=("mae", "mean"), rmse_mean=("rmse", "mean"), r2_mean=("r2", "mean"),
        r2_std=("r2", "std"), n_edges_mean=("n_edges", "mean")).reset_index())
    agg.to_csv(os.path.join(config.results_dir, f"step6_embedding_ablation_{args.tag}.csv"), index=False)
    print(agg.round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=LANGUAGE_MODEL_CHOICES, default="chemberta")
    parser.add_argument("--test-set", choices=("random", "hubs"), default="random",
                        help="random = Step 5's 15% random test set; hubs = Case Study 2's high-gravity hub test set")
    parser.add_argument("--parts", default="shap,columns,confidence,ablation", help="comma list of: shap, columns, confidence, ablation")
    parser.add_argument("--k", type=int, default=3, help="neighbours for the semantic graph in the ablation")
    parser.add_argument("--keep-dims", type=int, default=64, help="dimensions kept by the truncate_dims ablation")
    parser.add_argument("--shap-samples", type=int, default=500)
    parser.add_argument("--shap-min-leaf", type=int, default=5, help="min_samples_leaf of the forest SHAP explains")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2])
    args = parser.parse_args()
    parts = {p.strip() for p in args.parts.split(",")}

    config = RuntimeConfig(language_model=args.model)
    os.makedirs(config.results_dir, exist_ok=True)
    df = load_dataset("large")
    labels = df["pld_value"].values.astype(np.float32)
    X, meta = build_features(df, feature_scheme="fingerprint")
    E = cached_embed(df["linker_smiles"].tolist(), config).astype(np.float32)
    test_nodes = compute_global_fixed_test_nodes(len(df), config) if args.test_set == "random" else cs2_hub_test_nodes(df, config)
    args.tag = args.model + ("" if args.test_set == "random" else "_hubs")
    data_sha = hashlib.sha256(open(config.dataset_path, "rb").read()).hexdigest()[:16]
    base = meta_cols(args, config, data_sha, test_nodes)
    logger.info(f"X={X.shape} E={E.shape} test={len(test_nodes)} data_sha256[:16]={data_sha} parts={sorted(parts)}")
    t = time.time()
    if "shap" in parts:
        run_shap(args, config, X, E, meta, labels, test_nodes, base)
    if "columns" in parts:
        run_columns(args, config, X, labels, test_nodes, base)
    if "confidence" in parts:
        run_confidence(args, config, X, labels, test_nodes, base)
    if "ablation" in parts:
        run_ablation(args, config, df, X, E, labels, test_nodes, base)
    logger.info(f"Step 6 finished in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()
