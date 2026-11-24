#!/usr/bin/env python
"""Can the Random Forest be improved with standard, data-preserving steps? (original target, original features, no hand-made rules)

Stage A: grid search of max_features x min_samples_leaf (300 trees) scored on the VALIDATION split only (random test set splits,
         seeds 0-2); the test sets are never used to choose settings.
Stage B: on both test sets (random 15% and the 664-MOF Case Study 2 hub set), seeds 0-4, compare
         default forest | forest with the chosen settings | chosen settings trained on training + validation MOFs | ExtraTrees.

Usage:  PYTHONPATH=. python scripts/tune_random_forest.py
Writes  outputs/results/rf_tuning_validation_search.csv, outputs/results/rf_tuning_per_seed.csv
"""

import hashlib
import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.baseline.data_ingestion import load_dataset
from src.baseline.feature_engineering import build_features
from src.baseline.gnn_training import make_splits
from src.common.config import RuntimeConfig
from src.evaluation.run_comparison import compute_global_fixed_test_nodes

SCRIPT = "scripts/tune_random_forest.py"
GRID = list(itertools.product((0.3, 0.5, 1.0), (1, 2, 3)))  # (max_features, min_samples_leaf)


def main() -> None:
    config = RuntimeConfig(language_model="chemberta")
    df = load_dataset("large")
    y = df["pld_value"].values.astype(np.float32)
    X, _ = build_features(df, feature_scheme="fingerprint")
    sha = hashlib.sha256(open(config.dataset_path, "rb").read()).hexdigest()[:16]
    tests = {"random15": set(compute_global_fixed_test_nodes(len(df), config)),
             "cs2_hubs (664, repo code)": set(pd.read_csv(os.path.join(config.results_dir, "cs2_hub_test_nodes.csv")).node_index.tolist())}
    t0 = time.time()
    search = []
    for mf, msl in GRID:
        scores = []
        for seed in range(3):
            tr, va, _ = make_splits(len(X), tests["random15"], seed=seed)
            m = RandomForestRegressor(n_estimators=300, max_features=mf, min_samples_leaf=msl, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
            scores.append(r2_score(y[va], m.predict(X[va])))
        search.append({"max_features": mf, "min_samples_leaf": msl, "val_r2_mean": float(np.mean(scores)), "val_r2_std": float(np.std(scores)),
                       "script": SCRIPT, "data_sha256_16": sha})
        print(f"validation  max_features={mf} min_samples_leaf={msl}: R2 {np.mean(scores):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    sdf = pd.DataFrame(search).sort_values("val_r2_mean", ascending=False)
    sdf.to_csv(os.path.join(config.results_dir, "rf_tuning_validation_search.csv"), index=False)
    best = sdf.iloc[0]
    mf, msl = float(best.max_features), int(best.min_samples_leaf)
    print(f"chosen on validation: max_features={mf}, min_samples_leaf={msl}", flush=True)

    rows = []
    for tname, test in tests.items():
        te = np.array(sorted(test))
        for seed in range(5):
            tr, va, _ = make_splits(len(X), test, seed=seed)
            trva = np.concatenate([tr, va])

            def add(name, model, idx):
                p = model.fit(X[idx], y[idx]).predict(X[te])
                rows.append({"test_set": tname, "variant": name, "seed": seed, "r2": r2_score(y[te], p), "mae": mean_absolute_error(y[te], p),
                             "rmse": float(np.sqrt(mean_squared_error(y[te], p))), "n_train": len(idx), "n_test": len(te),
                             "script": SCRIPT, "data_sha256_16": sha})
            add("default Random Forest (100 trees)", RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1), tr)
            add(f"tuned Random Forest (300 trees, max_features={mf}, min_samples_leaf={msl})",
                RandomForestRegressor(n_estimators=300, max_features=mf, min_samples_leaf=msl, random_state=seed, n_jobs=-1), tr)
            add("tuned Random Forest, trained on training + validation",
                RandomForestRegressor(n_estimators=300, max_features=mf, min_samples_leaf=msl, random_state=seed, n_jobs=-1), trva)
            add("default Random Forest, trained on training + validation", RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1), trva)
            add("ExtraTrees (300 trees, default settings)", ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1), tr)
            print(tname, seed, f"{time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(config.results_dir, "rf_tuning_per_seed.csv"), index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    print(out.groupby(["test_set", "variant"])[["r2", "mae", "rmse"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
