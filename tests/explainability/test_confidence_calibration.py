import pytest

from src.explainability.confidence_calibration import calibrate_confidence


def test_calibrate_confidence_raises_until_final_model_selected():
    """Step 6 (confidence calibration) is stubbed until Step 5 has identified a final
    best-performing model configuration to calibrate. This test documents the current contract;
    replace with real assertions once calibrate_confidence is implemented."""
    with pytest.raises(NotImplementedError):
        calibrate_confidence(model=None, X_calibration=None, y_calibration=None)
