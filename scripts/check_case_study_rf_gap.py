#!/usr/bin/env python
"""Why is Case Study 2's Random Forest (R2 0.893, one seed) a little higher than the thesis's (0.888)?

Random Forest (100 trees, default settings) on the fingerprint features, tested on the 664-MOF Case Study 2 hub set that the repo
code selects (outputs/results/cs2_hub_test_nodes.csv, written by run_step6_explainability.py --test-set hubs), seeds 0-4, trained
either on the training split only (the validation MOFs held out, as in every thesis experiment) or on training + validation.

Usage:  PYTHONPATH=. python scripts/check_case_study_rf_gap.py
Writes  outputs/results/case_study_rf_gap_per_seed.csv
"""

import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.baseline.data_ingestion import load_dataset
from src.baseline.feature_engineering import build_features
from src.baseline.gnn_training import make_splits
from src.common.config import RuntimeConfig

SCRIPT = "scripts/check_case_study_rf_gap.py"


def main() -> None:
    config = RuntimeConfig(language_model="chemberta")
    df = load_dataset("large")
    y = df["pld_value"].values.astype(np.float32)
    X, _ = build_features(df, feature_scheme="fingerprint")
    hub = set(pd.read_csv(os.path.join(config.results_dir, "cs2_hub_test_nodes.csv")).node_index.tolist())
    te = np.array(sorted(hub))
    sha = hashlib.sha256(open(config.dataset_path, "rb").read()).hexdigest()[:16]
    rows = []
    for seed in range(5):
        tr, va, _ = make_splits(len(X), hub, seed=seed)
        for name, idx in (("training split only (validation held out)", tr), ("training + validation", np.concatenate([tr, va]))):
            model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1).fit(X[idx], y[idx])
            p = model.predict(X[te])
            rows.append({"training_data": name, "seed": seed, "r2": r2_score(y[te], p), "mae": mean_absolute_error(y[te], p),
                         "rmse": float(np.sqrt(mean_squared_error(y[te], p))), "n_test": len(te), "n_train": len(idx),
                         "script": SCRIPT, "data_sha256_16": sha, "test_set": "cs2_hubs (repo code, 664 MOFs)"})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(config.results_dir, "case_study_rf_gap_per_seed.csv"), index=False)
    print(out.groupby("training_data")[["r2", "mae", "rmse"]].agg(["mean", "std"]).round(4).T)
    print("Case Study 2 (Table 3, one seed): R2 0.893, MAE 0.525, RMSE 1.001")


if __name__ == "__main__":
    main()
