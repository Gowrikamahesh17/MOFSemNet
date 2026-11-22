#!/usr/bin/env python
"""Step 6d - SHAP on the language-model embedding dimensions, and what those dimensions encode.

A Random Forest on fingerprint + embedding features (min_samples_leaf=5, as in Step 6a) is explained with TreeSHAP over all
embedding dimensions. Because a dimension has no chemical meaning by itself, each important dimension (and the embedding group's
net contribution per MOF) is related to standard RDKit linker descriptors (weight, rings, polarity, ...) with Spearman correlations,
and a cross-validated ridge regression says how much of the embedding contribution those descriptors explain.

Usage:  PYTHONPATH=. python scripts/run_step6d_embedding_shap.py [--models chemberta molt5] [--n-explain 500]
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd

from src.baseline.data_ingestion import load_dataset
from src.baseline.feature_engineering import build_features
from src.baseline.gnn_training import make_splits
from src.common.config import RuntimeConfig
from src.common.logging_setup import get_logger
from src.evaluation.run_comparison import compute_global_fixed_test_nodes
from src.explainability.shap_analysis import compute_shap_values, feature_group, fingerprint_feature_names
from src.language.embeddings import cached_embed

logger = get_logger(__name__)
SCRIPT = "scripts/run_step6d_embedding_shap.py"
UI_DIR = os.path.join("frontend", "public", "step6")
DESCRIPTORS = ["molecular weight", "heavy atoms", "rings", "aromatic rings", "rotatable bonds", "TPSA", "logP", "fraction sp3",
               "H-bond donors", "H-bond acceptors", "heteroatoms", "formal charge"]


def linker_descriptors(smiles_unique):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    rows = []
    for smi in smiles_unique:
        m = Chem.MolFromSmiles(smi)
        if m is None or m.GetNumAtoms() == 0:
            rows.append([np.nan] * len(DESCRIPTORS))
            continue
        rows.append([Descriptors.MolWt(m), m.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumRings(m), rdMolDescriptors.CalcNumAromaticRings(m),
                     rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcTPSA(m), Descriptors.MolLogP(m),
                     rdMolDescriptors.CalcFractionCSP3(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
                     rdMolDescriptors.CalcNumHeteroatoms(m), Chem.GetFormalCharge(m)])
    arr = np.array(rows, dtype=float)
    return np.where(np.isnan(arr), np.nanmedian(arr, axis=0)[None, :], arr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["chemberta", "molt5"])
    ap.add_argument("--n-explain", type=int, default=500)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    from scipy.stats import spearmanr
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import r2_score

    config = RuntimeConfig(language_model="chemberta")
    df = load_dataset("large")
    y = df["pld_value"].values.astype(np.float32)
    X, meta = build_features(df, feature_scheme="fingerprint")
    test_nodes = compute_global_fixed_test_nodes(len(df), config)
    tr, _va, _ = make_splits(len(X), test_nodes, seed=0)
    te = np.array(sorted(test_nodes))
    data_sha = hashlib.sha256(open(config.dataset_path, "rb").read()).hexdigest()[:16]
    base = {"script": SCRIPT, "data_sha256_16": data_sha, "split_random_state": config.split_random_state, "seed": 0}
    uniq, inv = np.unique(df["linker_smiles"].values, return_inverse=True)
    D = linker_descriptors(uniq)[inv]  # per MOF
    summary, corr_all, dim_all = [], [], []
    for lm in args.models:
        E = cached_embed(df["linker_smiles"].tolist(), RuntimeConfig(language_model=lm)).astype(np.float32)
        M = np.hstack([X, E]).astype(np.float32)
        names = fingerprint_feature_names(meta, E.shape[1])
        groups = np.array([feature_group(n) for n in names])
        rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1, min_samples_leaf=5).fit(M[tr], y[tr])
        test_r2 = float(r2_score(y[te], rf.predict(M[te])))
        out = compute_shap_values(rf, M[te], names, max_samples=args.n_explain, seed=0)
        S, rows = out["shap_values"], te[out["row_index"]]
        emb_cols = np.where(groups == "language-model embedding")[0]
        mean_abs = np.abs(S[:, emb_cols]).mean(0)
        order = np.argsort(-mean_abs)
        net = S[:, emb_cols].sum(axis=1)  # embedding group's net contribution per explained MOF
        logger.info(f"{lm}: test R2 {test_r2:.3f}; embedding group net share {out['groups']['language-model embedding']['net_share']:.1%}, "
                    f"summed-column share {out['groups']['language-model embedding']['share']:.1%}; top-{args.top} dims carry "
                    f"{mean_abs[order[:args.top]].sum() / mean_abs.sum():.1%} of the summed embedding importance")
        for rank, j in enumerate(order[:args.top], start=1):
            col = emb_cols[j]
            rho = [spearmanr(E[:, j], D[:, d])[0] for d in range(len(DESCRIPTORS))]
            best = int(np.nanargmax(np.abs(rho)))
            dim_all.append({"language_model": lm, "rank": rank, "dimension": int(j), "mean_abs_shap": float(mean_abs[j]),
                            "share_of_embedding_importance": float(mean_abs[j] / mean_abs.sum()),
                            "best_matching_descriptor": DESCRIPTORS[best], "spearman_with_best": float(rho[best]),
                            **{f"spearman_{d}": float(r) for d, r in zip(DESCRIPTORS, rho)}, **base})
        for d, name in enumerate(DESCRIPTORS):
            r_, _ = spearmanr(net, D[rows, d])
            corr_all.append({"language_model": lm, "descriptor": name, "spearman_with_net_embedding_shap": float(r_), **base})
        pred = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), (D[rows] - D[rows].mean(0)) / np.where(D[rows].std(0) == 0, 1, D[rows].std(0)), net, cv=5)
        r2_desc = float(r2_score(net, pred))
        # does the embedding's contribution just recover pore size? correlate with the true PLD too
        r_pld = float(spearmanr(net, y[rows])[0])
        summary.append({"language_model": lm, "test_r2": test_r2, "embedding_share_sum_of_columns": out["groups"]["language-model embedding"]["share"],
                        "embedding_share_net": out["groups"]["language-model embedding"]["net_share"],
                        "top_dims_share_of_embedding_importance": float(mean_abs[order[:args.top]].sum() / mean_abs.sum()),
                        "r2_net_embedding_shap_from_descriptors_cv": r2_desc, "spearman_net_embedding_shap_vs_true_pld": r_pld,
                        "n_explained": len(rows), **base})
        logger.info(f"{lm}: descriptors explain R2 {r2_desc:.2f} of the embedding's net contribution; Spearman with true PLD {r_pld:.2f}")
    res = config.results_dir
    pd.DataFrame(summary).to_csv(os.path.join(res, "step6d_embedding_shap_summary.csv"), index=False)
    pd.DataFrame(dim_all).to_csv(os.path.join(res, "step6d_embedding_top_dimensions.csv"), index=False)
    pd.DataFrame(corr_all).to_csv(os.path.join(res, "step6d_embedding_net_vs_descriptors.csv"), index=False)
    print(pd.DataFrame(summary).round(3).to_string())
    print(pd.DataFrame(dim_all)[["language_model", "rank", "dimension", "share_of_embedding_importance", "best_matching_descriptor", "spearman_with_best"]].round(3).to_string())
    make_figure(config, pd.DataFrame(dim_all), pd.DataFrame(corr_all))


def make_figure(config, dims, corr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lms = list(dict.fromkeys(dims["language_model"]))
    fig, axes = plt.subplots(1, len(lms), figsize=(6.2 * len(lms), 5.4), squeeze=False)
    for ax, lm in zip(axes[0], lms):
        d = dims[dims.language_model == lm]
        mat = d[[f"spearman_{x}" for x in DESCRIPTORS]].values
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(DESCRIPTORS)))
        ax.set_xticklabels(DESCRIPTORS, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([f"dim {r.dimension} ({r.share_of_embedding_importance:.1%})" for r in d.itertuples()], fontsize=7)
        ax.set_title(f"{lm}: top SHAP dimensions vs linker descriptors (Spearman)", fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    os.makedirs(config.figures_dir, exist_ok=True)
    os.makedirs(UI_DIR, exist_ok=True)
    fig.savefig(os.path.join(config.figures_dir, "step6d_embedding_dimensions.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(UI_DIR, "embedding_dimensions.png"), dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
