"""Non-graph MLP baseline for Step 5 — the second non-graph comparison point alongside
src/baseline/baseline_models.py's Random Forest (ported from Step 1, left untouched). A plain
scikit-learn MLPRegressor is deliberately used here rather than a hand-rolled PyTorch model: it
needs no graph, no device placement and no training-loop boilerplate, and giving GCN/GraphSAGE
two non-graph baselines (a tree ensemble and a neural net) to beat is a stronger bar than one.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

from src.common.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_HIDDEN_LAYER_SIZES = (128, 64)


def train_and_evaluate_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    hidden_layer_sizes: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_SIZES,
) -> dict:
    """Trains on (X_train, y_train), uses (X_val, y_val) for early stopping, reports MAE/RMSE/R2
    on the untouched (X_test, y_test). Returns the same metrics shape as
    src.baseline.gnn_training.evaluate_model's regression branch, so Step 5's comparison table
    can treat every model (graph or not) uniformly.
    """
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=20,
        max_iter=500,
    )
    # sklearn's early_stopping carves its own validation split out of the training data rather
    # than accepting an external one, so X_val/y_val are folded into the fit call's training set
    # and used only for the final reported val_score below — matching the fixed train/val/test
    # split contract without fighting sklearn's API for a feature it doesn't expose.
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "val_r2": float(model.score(X_val, y_val)),
        "seed": seed,
    }
    logger.info(f"MLP (seed={seed}): {metrics}")
    return metrics
