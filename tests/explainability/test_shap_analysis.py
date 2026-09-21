import pytest

from src.explainability.shap_analysis import compute_shap_values


def test_compute_shap_values_raises_until_step5_models_exist():
    """Step 6 (SHAP analysis) is stubbed until Step 5 has produced a trained baseline model to
    explain. This test documents the current contract; replace with real assertions once
    compute_shap_values is implemented."""
    with pytest.raises(NotImplementedError):
        compute_shap_values(model=None, X=None, feature_names=[])
