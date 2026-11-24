#!/usr/bin/env python
"""Standard non-graph models beyond the Random Forest, all tuned the same way (original data, original target, no hand-made rules).

Models: Random Forest (tuned, reference), ExtraTrees, XGBoost, LightGBM, CatBoost, HistGradientBoosting, a Tanimoto-kernel regressor
(= the mean of a Gaussian process; kernel = Tanimoto on the fingerprint bits + RBF on the two pore columns + same-metal term),
Ridge regression (linear reference), and two ensembles of the tuned models.

Protocol
  Stage A  settings of every model are chosen on the VALIDATION split only (random-test-set splits, seeds 0-1); the test sets are
           never used to choose anything. Random search spaces are drawn with a fixed generator, so runs are repeatable.
  Stage B  each model with its chosen settings is trained on the training split (validation MOFs held out, as in every thesis
           experiment) and tested on the random 15% test set and on the 664-MOF Case Study 2 hub set, seeds 0-4.
  Ensembles: equal average of the 3 models with the best validation R2 in that seed, and non-negative least-squares weights fitted on
           the validation predictions (normalised to sum 1). Weights never see the test set.

Usage:  PYTHONPATH=. python scripts/run_model_zoo.py [--models et,xgboost,...] [--quick]
Writes  outputs/results/model_zoo_validation_search.csv, outputs/results/model_zoo_per_seed.csv
"""

import argparse
import hashlib
import itertools
import os
import time

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.baseline.data_ingestion import load_dataset
from src.baseline.feature_engineering import build_features
from src.baseline.gnn_training import make_splits
from src.common.config import RuntimeConfig
from src.evaluation.run_comparison import compute_global_fixed_test_nodes

SCRIPT = "scripts/run_model_zoo.py"
N_FP = 1024
ALL_MODELS = ["random_forest", "extra_trees", "xgboost", "lightgbm", "catboost", "hist_gradient_boosting", "tanimoto_kernel", "ridge"]


# ------------------------------------------------------------------------------------------------ search spaces
def sample(space: dict, n: int, seed: int = 0) -> list[dict]:
    combos = [dict(zip(space, v)) for v in itertools.product(*space.values())]
    rng = np.random.default_rng(seed)
    if len(combos) <= n:
        return combos
    return [combos[i] for i in rng.choice(len(combos), size=n, replace=False)]


SPACES = {
    "random_forest": ({"max_features": (0.3, 0.5, 1.0), "min_samples_leaf": (1, 2)}, 6),
    "extra_trees": ({"max_features": (0.3, 0.5, 1.0), "min_samples_leaf": (1, 2)}, 6),
    "xgboost": ({"max_depth": (4, 6, 8, 10), "learning_rate": (0.03, 0.05, 0.1), "subsample": (0.7, 0.9), "colsample_bytree": (0.2, 0.3, 0.5),
                 "min_child_weight": (1, 3, 5), "reg_lambda": (1.0, 5.0)}, 12),
    "lightgbm": ({"num_leaves": (15, 31, 63, 127), "learning_rate": (0.03, 0.05, 0.1), "colsample_bytree": (0.2, 0.3, 0.5), "subsample": (0.7, 0.9),
                  "min_child_samples": (5, 10, 20), "reg_lambda": (0.0, 5.0)}, 12),
    "catboost": ({"depth": (6, 8), "learning_rate": (0.05, 0.1), "l2_leaf_reg": (3.0, 10.0)}, 4),
    "hist_gradient_boosting": ({"max_leaf_nodes": (31, 63, 127), "learning_rate": (0.05, 0.1), "l2_regularization": (0.0, 1.0), "min_samples_leaf": (10, 20)}, 8),
    "tanimoto_kernel": ({"w_pore": (2.0, 6.0), "gamma": (0.5, 2.0), "w_metal": (0.2, 1.0), "noise": (0.03, 0.1)}, 8),
    "ridge": ({"dummy": (0,)}, 1),
}


# ------------------------------------------------------------------------------------------------ models
class Data:
    def __init__(self, X, y):
        self.X, self.y = X, y
        self.fp, self.pore, self.metal = X[:, :N_FP], X[:, N_FP:N_FP + 2], X[:, N_FP + 2:]


def tanimoto(a, b):
    inter = a @ b.T
    return inter / np.maximum(a.sum(1)[:, None] + b.sum(1)[None, :] - inter, 1.0)


def kernel(d: Data, ia, ib, p, mu, sd):
    pa, pb = (d.pore[ia] - mu) / sd, (d.pore[ib] - mu) / sd
    d2 = (pa ** 2).sum(1)[:, None] + (pb ** 2).sum(1)[None, :] - 2 * pa @ pb.T
    return tanimoto(d.fp[ia], d.fp[ib]) + p["w_pore"] * np.exp(-p["gamma"] * np.maximum(d2, 0)) + p["w_metal"] * (d.metal[ia] @ d.metal[ib].T)


def fit_predict(name, p, d: Data, tr, va, te, seed):
    """Returns (validation predictions, test predictions) of a model trained on `tr` only (early stopping, where used, looks at `va`)."""
    X, y = d.X, d.y
    if name in ("random_forest", "extra_trees"):
        cls = RandomForestRegressor if name == "random_forest" else ExtraTreesRegressor
        m = cls(n_estimators=300, random_state=seed, n_jobs=-1, **p).fit(X[tr], y[tr])
        return m.predict(X[va]), m.predict(X[te])
    if name == "xgboost":
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=2000, tree_method="hist", early_stopping_rounds=50, random_state=seed, n_jobs=-1, verbosity=0, **p)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        return m.predict(X[va]), m.predict(X[te])
    if name == "lightgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=2000, subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1, **p)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
        return m.predict(X[va]), m.predict(X[te])
    if name == "catboost":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=1500, rsm=0.3, random_seed=seed, thread_count=-1, verbose=0, early_stopping_rounds=100, **p)
        m.fit(X[tr], y[tr], eval_set=(X[va], y[va]))
        return m.predict(X[va]), m.predict(X[te])
    if name == "hist_gradient_boosting":
        m = HistGradientBoostingRegressor(max_iter=1000, early_stopping=True, n_iter_no_change=30, random_state=seed, **p).fit(X[tr], y[tr])
        return m.predict(X[va]), m.predict(X[te])
    if name == "tanimoto_kernel":
        mu, sd = d.pore[tr].mean(0), d.pore[tr].std(0) + 1e-9
        ym = y[tr].mean()
        K = kernel(d, tr, tr, p, mu, sd)
        K[np.diag_indices_from(K)] += p["noise"]
        alpha = cho_solve(cho_factor(K, overwrite_a=True, check_finite=False), y[tr] - ym, check_finite=False)
        del K
        return kernel(d, va, tr, p, mu, sd) @ alpha + ym, kernel(d, te, tr, p, mu, sd) @ alpha + ym
    if name == "ridge":
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), y[tr])
        return m.predict(sc.transform(X[va])), m.predict(sc.transform(X[te]))
    raise ValueError(name)


def metrics(y, p):
    return {"r2": float(r2_score(y, p)), "mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p)))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 configs per model, 1 seed, 1 test set")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",")]

    config = RuntimeConfig(language_model="chemberta")
    df = load_dataset("large")
    y = df["pld_value"].values.astype(np.float64)
    X, _ = build_features(df, feature_scheme="fingerprint")
    X = X.astype(np.float32)
    d = Data(X, y)
    sha = hashlib.sha256(open(config.dataset_path, "rb").read()).hexdigest()[:16]
    tests = {"random15": set(compute_global_fixed_test_nodes(len(df), config)),
             "cs2_hubs (664, repo code)": set(pd.read_csv(os.path.join(config.results_dir, "cs2_hub_test_nodes.csv")).node_index.tolist())}
    sel_seeds, test_seeds = ([0], [0]) if args.quick else ([0, 1], list(range(5)))
    if args.quick:
        tests = {k: v for k, v in list(tests.items())[:1]}
    t0 = time.time()

    # ------------------------------------------------ Stage A: choose settings on the validation split only
    chosen, search_rows = {}, []
    for name in models:
        space, n = SPACES[name]
        configs = sample(space, 2 if args.quick else n)
        best = None
        for p in configs:
            scores = []
            for seed in sel_seeds:
                tr, va, _ = make_splits(len(X), tests["random15"], seed=seed)
                pv, _ = fit_predict(name, p, d, tr, va, va[:5], seed)
                scores.append(r2_score(y[va], pv))
            row = {"model": name, "settings": str(p), "val_r2_mean": float(np.mean(scores)), "val_r2_std": float(np.std(scores)), "script": SCRIPT, "data_sha256_16": sha}
            search_rows.append(row)
            print(f"[{time.time() - t0:5.0f}s] validation {name} {p}: R2 {np.mean(scores):.4f}", flush=True)
            if best is None or row["val_r2_mean"] > best[0]:
                best = (row["val_r2_mean"], p)
        chosen[name] = best[1]
        print(f"[{time.time() - t0:5.0f}s] CHOSEN {name}: {best[1]}  (validation R2 {best[0]:.4f})", flush=True)
        pd.DataFrame(search_rows).to_csv(os.path.join(config.results_dir, "model_zoo_validation_search.csv"), index=False)

    # ------------------------------------------------ Stage B: test with the chosen settings
    rows = []
    for tname, test in tests.items():
        te = np.array(sorted(test))
        for seed in test_seeds:
            tr, va, _ = make_splits(len(X), test, seed=seed)
            preds = {}
            for name in models:
                pv, pt = fit_predict(name, chosen[name], d, tr, va, te, seed)
                preds[name] = (pv, pt)
                rows.append({"test_set": tname, "model": name, "settings": str(chosen[name]), "seed": seed, "val_r2": r2_score(y[va], pv), **metrics(y[te], pt),
                             "n_train": len(tr), "n_test": len(te), "script": SCRIPT, "data_sha256_16": sha})
                print(f"[{time.time() - t0:5.0f}s] {tname} seed {seed} {name}: test R2 {rows[-1]['r2']:.4f}", flush=True)
            if len(models) >= 3:  # ensembles use validation predictions only
                names = list(preds)
                V, T = np.stack([preds[n][0] for n in names], 1), np.stack([preds[n][1] for n in names], 1)
                top = np.argsort([-r2_score(y[va], V[:, i]) for i in range(len(names))])[:3]
                w_nnls, _ = nnls(V, y[va])
                w_nnls = w_nnls / w_nnls.sum() if w_nnls.sum() > 0 else np.ones(len(names)) / len(names)
                for ename, w in (("ensemble: average of the 3 best on validation", np.isin(np.arange(len(names)), top) / 3.0),
                                 ("ensemble: validation-weighted (NNLS)", w_nnls)):
                    rows.append({"test_set": tname, "model": ename, "settings": ", ".join(f"{n}={x:.2f}" for n, x in zip(names, w) if x > 0), "seed": seed,
                                 "val_r2": r2_score(y[va], V @ w), **metrics(y[te], T @ w), "n_train": len(tr), "n_test": len(te), "script": SCRIPT, "data_sha256_16": sha})
            pd.DataFrame(rows).to_csv(os.path.join(config.results_dir, "model_zoo_per_seed.csv"), index=False)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(out.groupby(["test_set", "model"])[["r2", "mae", "rmse"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
