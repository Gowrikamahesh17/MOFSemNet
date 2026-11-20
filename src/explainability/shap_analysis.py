"""Step 6a: SHAP analysis for the non-graph baseline (Random Forest, the best model in Step 5).

SHAP assigns each input column a signed contribution to every prediction; the mean of the absolute
values over many MOFs says how much the model relies on that column. Columns are also summed into
groups (linker fingerprint bits, pore geometry, metal, language-model embedding) so the thesis can
state which kind of information the model actually uses.
"""

from __future__ import annotations

import numpy as np


def fingerprint_feature_names(metadata: dict, embedding_dim: int = 0) -> list[str]:
    """Column names matching src.baseline.feature_engineering.build_fingerprint_features:
    Morgan bits, then the two pore-geometry columns, then the metal one-hot (ordered by the
    index stored in `metadata`). `embedding_dim` > 0 appends names for language-model columns."""
    names = [f"fp_bit_{i}" for i in range(metadata["morgan_bits"])]
    names += [f"pore_{c}" for c in metadata["pore_geometry_columns"]]
    metals = sorted(metadata["metal_to_index"], key=metadata["metal_to_index"].get)
    names += [f"metal_{m}" for m in metals]
    names += [f"emb_{i}" for i in range(embedding_dim)]
    return names


def feature_group(name: str) -> str:
    if name.startswith("fp_bit_"):
        return "linker fingerprint"
    if name.startswith("pore_"):
        return "pore geometry"
    if name.startswith("metal_"):
        return "metal"
    if name.startswith("emb_"):
        return "language-model embedding"
    return "other"


def group_importance(mean_abs_shap: dict[str, float]) -> dict[str, dict]:
    """Sum of mean |SHAP| per feature group, with each group's share of the total."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, value in mean_abs_shap.items():
        g = feature_group(name)
        totals[g] = totals.get(g, 0.0) + float(value)
        counts[g] = counts.get(g, 0) + 1
    grand = sum(totals.values()) or 1.0
    return {g: {"n_features": counts[g], "mean_abs_shap_sum": totals[g], "share": totals[g] / grand}
            for g in sorted(totals, key=totals.get, reverse=True)}


def compute_shap_values(model, X: np.ndarray, feature_names: list[str], max_samples: int = 500, seed: int = 0,
                        additivity_tolerance: float = 0.05) -> dict:
    """Returns {"shap_values": (n_used, n_features) array, "feature_names": [...],
    "mean_abs_shap": {feature_name: importance}, "row_index": indices of the rows explained,
    "groups": group_importance(...)} for a fitted non-graph baseline.

    Tree ensembles (RandomForestRegressor and similar) use shap.TreeExplainer (exact). Any other
    model with a `.predict` uses shap.KernelExplainer on at most 100 rows and a 50-row background,
    which is approximate and slow for ~1,000 columns.

    Safety check: SHAP values must add up to the prediction (base value + sum of contributions). TreeSHAP
    loses this property on fully grown forests (unlimited depth, one sample per leaf), where it can be off by
    thousands, so the largest gap is returned as "max_additivity_error" and a ValueError is raised if it
    exceeds `additivity_tolerance` (in the units of the target). Fit the forest with min_samples_leaf >= 3 or a
    max_depth to avoid this.
    """
    import shap

    if X.shape[1] != len(feature_names):
        raise ValueError(f"X has {X.shape[1]} columns but {len(feature_names)} feature names were given")

    rng = np.random.default_rng(seed)
    is_tree = hasattr(model, "estimators_")
    n_used = min(len(X), max_samples if is_tree else min(max_samples, 100))
    rows = np.sort(rng.choice(len(X), size=n_used, replace=False))
    X_used = X[rows]

    if is_tree:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_used, check_additivity=False)
        base = float(np.ravel(explainer.expected_value)[0])
    else:
        background = shap.sample(X, min(50, len(X)), random_state=seed)
        explainer = shap.KernelExplainer(model.predict, background)
        values = explainer.shap_values(X_used, nsamples=200)
        base = float(np.ravel(explainer.expected_value)[0])
    values = np.asarray(values)

    gap = float(np.abs(values.sum(axis=1) + base - model.predict(X_used)).max())
    if gap > additivity_tolerance:
        raise ValueError(
            f"SHAP values do not add up to the predictions (largest gap {gap:.3f}, tolerance {additivity_tolerance}). "
            "TreeSHAP is unreliable on fully grown forests; refit with min_samples_leaf >= 3 or a max_depth.")

    mean_abs = {n: float(v) for n, v in zip(feature_names, np.abs(values).mean(axis=0))}
    return {"shap_values": values, "feature_names": list(feature_names), "mean_abs_shap": mean_abs,
            "row_index": rows, "groups": group_importance(mean_abs), "max_additivity_error": gap}
