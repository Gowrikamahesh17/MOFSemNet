import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.explainability.shap_analysis import (compute_shap_values, feature_group, fingerprint_feature_names,
                                              group_importance)


def _fitted_forest():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6)).astype(np.float32)
    y = 3 * X[:, 0] + 0.1 * rng.normal(size=300)  # only column 0 matters
    return RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y), X


def test_shap_ranks_the_informative_column_first():
    model, X = _fitted_forest()
    names = [f"fp_bit_{i}" for i in range(6)]
    out = compute_shap_values(model, X, names, max_samples=100)
    assert out["shap_values"].shape == (100, 6)
    assert max(out["mean_abs_shap"], key=out["mean_abs_shap"].get) == "fp_bit_0"


def test_shap_rejects_name_count_mismatch():
    model, X = _fitted_forest()
    with pytest.raises(ValueError):
        compute_shap_values(model, X, ["a", "b"])


def test_feature_names_and_groups():
    meta = {"morgan_bits": 3, "pore_geometry_columns": ["LCD", "LFS"], "metal_to_index": {"Zn": 0, "Cu": 1}}
    names = fingerprint_feature_names(meta, embedding_dim=2)
    assert names == ["fp_bit_0", "fp_bit_1", "fp_bit_2", "pore_LCD", "pore_LFS", "metal_Zn", "metal_Cu", "emb_0", "emb_1"]
    assert [feature_group(n) for n in names][3] == "pore geometry"
    groups = group_importance({n: 1.0 for n in names})
    assert groups["linker fingerprint"]["n_features"] == 3
    assert abs(sum(g["share"] for g in groups.values()) - 1.0) < 1e-9


def test_additivity_guard_reports_a_small_gap_and_rejects_a_bad_one():
    model, X = _fitted_forest()
    names = [f"fp_bit_{i}" for i in range(6)]
    out = compute_shap_values(model, X, names, max_samples=50)
    assert out["max_additivity_error"] < 1e-3
    with pytest.raises(ValueError):
        compute_shap_values(model, X, names, max_samples=50, additivity_tolerance=-1.0)
