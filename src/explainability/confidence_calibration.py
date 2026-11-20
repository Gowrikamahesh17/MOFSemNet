"""Step 6c: calibrated confidence for the final model (Random Forest) via split conformal prediction.

A point prediction becomes an interval that, on new MOFs drawn the same way as the calibration set,
contains the true PLD with probability at least 1 - alpha. The calibration set must be disjoint from
both the training set and the final test set (Step 5's validation split is used for this).

Two interval types:
  - "absolute":   one width for every MOF (residual quantile).
  - "normalized": width proportional to the spread of the forest's tree predictions, so MOFs the trees
                  disagree about get wider intervals. This is what makes the interval a usable
                  per-MOF confidence signal.
"""

from __future__ import annotations

import math

import numpy as np

_EPS = 1e-3


def _scale(model, X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "absolute":
        return np.ones(len(X))
    if mode == "normalized":
        if not hasattr(model, "estimators_"):
            raise ValueError("mode='normalized' needs a tree ensemble (an object with .estimators_)")
        per_tree = np.stack([t.predict(X) for t in model.estimators_])
        return per_tree.std(axis=0) + _EPS
    raise ValueError(f"Unknown mode {mode!r}; use 'absolute' or 'normalized'")


def calibrate_confidence(model, X_calibration: np.ndarray, y_calibration: np.ndarray, alpha: float = 0.1,
                         mode: str = "absolute") -> dict:
    """Fit the conformal quantile on the calibration split. Returns
    {"q_hat": float, "alpha": alpha, "mode": mode, "n_calibration": n}; pass it to predict_interval."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    n = len(X_calibration)
    if n < 2:
        raise ValueError("Need at least 2 calibration points")
    scores = np.abs(y_calibration - model.predict(X_calibration)) / _scale(model, X_calibration, mode)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)  # finite-sample correction
    return {"q_hat": float(np.quantile(scores, level, method="higher")), "alpha": alpha, "mode": mode,
            "n_calibration": n}


def predict_interval(model, X: np.ndarray, calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) arrays for each row of X."""
    pred = model.predict(X)
    half = calibration["q_hat"] * _scale(model, X, calibration["mode"])
    return pred - half, pred + half


def evaluate_coverage(model, X_test: np.ndarray, y_test: np.ndarray, calibration: dict) -> dict:
    """Empirical coverage (fraction of true values inside the interval) and interval width on the test set."""
    lower, upper = predict_interval(model, X_test, calibration)
    inside = (y_test >= lower) & (y_test <= upper)
    width = upper - lower
    return {"coverage": float(inside.mean()), "target": 1 - calibration["alpha"], "mean_width": float(width.mean()),
            "median_width": float(np.median(width)), "inside": inside, "width": width}
