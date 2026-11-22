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


def test_bit_substructures_maps_bit_to_fragment(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from src.explainability.shap_analysis import bit_substructures

    smiles = ["OC(=O)c1ccccc1", "OC(=O)c1ccc(cc1)C(=O)O", "c1ccccc1"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    bits = [int(i) for i in gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles[0])).nonzero()[0][:3]]
    out = bit_substructures(smiles, bits, svg_dir=str(tmp_path))
    assert [o["bit"] for o in out] == bits
    for o in out:
        assert o["fragment_smiles"] and o["n_linkers_with_bit"] >= 1 and 0 < o["fragment_share"] <= 1
        assert (tmp_path / o["svg"]).read_text().startswith("<?xml") or "<svg" in (tmp_path / o["svg"]).read_text()


def test_group_shares_sum_to_one():
    import numpy as np
    from src.explainability.model_explainers import group_shares
    v = np.array([[1.0, -1.0, 2.0], [1.0, 1.0, -2.0]])
    shares = group_shares(v, ["a", "a", "b"])
    assert abs(sum(shares.values()) - 1.0) < 1e-9 and shares["b"] == 0.5


def test_xgboost_shap_is_additive():
    import numpy as np
    import xgboost as xgb
    from src.explainability.shap_analysis import compute_shap_values
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6)).astype(np.float32)
    y = (2 * X[:, 0] + X[:, 1]).astype(np.float32)
    m = xgb.XGBRegressor(n_estimators=40, max_depth=3, base_score=float(y.mean()), verbosity=0).fit(X, y)
    out = compute_shap_values(m, X, [f"f{i}" for i in range(6)], max_samples=100)
    assert out["max_additivity_error"] < 1e-3
    assert max(out["mean_abs_shap"], key=out["mean_abs_shap"].get) == "f0"


def test_graph_predict_fn_matches_direct_forward():
    import numpy as np
    import torch
    from src.explainability.model_explainers import _build_model, graph_predict_fn
    torch.manual_seed(0)
    n, f = 12, 5
    X = np.random.default_rng(0).normal(size=(n, f)).astype(np.float32)
    ei = np.array([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
    model = _build_model("gcn", f).eval()
    fn = graph_predict_fn(model, X, ei, None, node=2, hops=2, y_mean=0.0, y_std=1.0)
    direct = model(torch.tensor(X), torch.tensor(ei)).squeeze(-1)[2].item()
    assert abs(fn(X[2:3])[0] - direct) < 1e-4
    other = X[2:3] + 1.0
    assert abs(fn(other)[0] - direct) > 1e-6
