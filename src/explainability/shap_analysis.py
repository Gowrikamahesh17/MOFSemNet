"""Step 6a: SHAP analysis for the non-graph baseline (Random Forest — src.baseline.baseline_models
/ src.models.mlp). Stubbed until Step 5 has produced a trained baseline model to explain; the
signature and return shape are fixed now so src/evaluation and reports/ can be written against a
stable contract.
"""

from __future__ import annotations

import numpy as np


def compute_shap_values(model, X: np.ndarray, feature_names: list[str]) -> dict:
    """Returns {"shap_values": (n_samples, n_features) array, "feature_names": [...],
    "mean_abs_shap": {feature_name: importance}} for the fitted non-graph baseline `model`
    (a RandomForestRegressor or MLPRegressor).

    Not yet implemented — Step 6 depends on Step 5's trained models existing first. Will use
    shap.TreeExplainer for Random Forest and shap.KernelExplainer (or shap.Explainer's
    model-agnostic default) for the MLP.
    """
    raise NotImplementedError(
        "compute_shap_values: Step 6 depends on a trained Step 5 baseline model. "
        "Implement once src.evaluation.run_comparison has produced one to explain."
    )
