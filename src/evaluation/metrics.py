"""Step 5: aggregate per-seed metrics (MAE/RMSE/R2 from src.baseline.gnn_training.evaluate_model
or src.models.mlp.train_and_evaluate_mlp) into the mean +/- std table the thesis exposé asks for,
reported per (model, graph type, sparsification method, retention level) configuration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_NAMES = ("mae", "rmse", "r2")


def aggregate_seed_runs(per_seed_metrics: list[dict]) -> dict:
    """`per_seed_metrics` is a list of {"mae": ..., "rmse": ..., "r2": ..., "seed": ...} dicts,
    one per seed, for a single (model, graph type, sparsification, retention) configuration.
    Returns {"mae_mean", "mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std", "n_seeds"}.
    """
    if not per_seed_metrics:
        raise ValueError("per_seed_metrics is empty — nothing to aggregate.")

    result = {"n_seeds": len(per_seed_metrics)}
    for metric in METRIC_NAMES:
        values = np.array([run[metric] for run in per_seed_metrics], dtype=np.float64)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std(ddof=0))
    return result


def build_comparison_table(configurations: dict[tuple, list[dict]]) -> pd.DataFrame:
    """`configurations` maps a config key (e.g. ("GCN", "semantic", "black_hole", 0.3)) to its
    list of per-seed metric dicts. Returns one row per configuration with aggregated
    mean +/- std columns, ready to write to outputs/results/.
    """
    rows = []
    for key, per_seed_metrics in configurations.items():
        aggregated = aggregate_seed_runs(per_seed_metrics)
        row = dict(zip(("model", "graph_type", "sparsification_method", "retention_level"), key))
        row.update(aggregated)
        rows.append(row)
    return pd.DataFrame(rows)
