import numpy as np

from src.models.mlp import train_and_evaluate_mlp


def test_mlp_reports_expected_metric_keys():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=60)  # learnable linear-ish signal

    metrics = train_and_evaluate_mlp(
        X[:30], y[:30], X[30:45], y[30:45], X[45:], y[45:], seed=0, hidden_layer_sizes=(8,)
    )
    assert {"mae", "rmse", "r2", "val_r2", "seed"}.issubset(metrics.keys())
    assert metrics["seed"] == 0
