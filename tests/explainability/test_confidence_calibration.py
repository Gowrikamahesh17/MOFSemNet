import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.explainability.confidence_calibration import calibrate_confidence, evaluate_coverage, predict_interval


def _data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    return X, X[:, 0] + 0.5 * rng.normal(size=n)


@pytest.mark.parametrize("mode", ["absolute", "normalized"])
def test_coverage_is_close_to_target(mode):
    Xtr, ytr = _data(600, 0)
    Xcal, ycal = _data(800, 1)
    Xte, yte = _data(2000, 2)
    model = RandomForestRegressor(n_estimators=40, random_state=0).fit(Xtr, ytr)
    cal = calibrate_confidence(model, Xcal, ycal, alpha=0.1, mode=mode)
    result = evaluate_coverage(model, Xte, yte, cal)
    assert 0.86 <= result["coverage"] <= 0.94
    lower, upper = predict_interval(model, Xte, cal)
    assert (upper > lower).all()


def test_normalized_widths_vary_and_absolute_widths_do_not():
    Xtr, ytr = _data(400, 0)
    Xcal, ycal = _data(400, 1)
    model = RandomForestRegressor(n_estimators=40, random_state=0).fit(Xtr, ytr)
    wa = evaluate_coverage(model, Xcal, ycal, calibrate_confidence(model, Xcal, ycal, mode="absolute"))["width"]
    wn = evaluate_coverage(model, Xcal, ycal, calibrate_confidence(model, Xcal, ycal, mode="normalized"))["width"]
    assert np.ptp(wa) < 1e-9 and np.ptp(wn) > 0.01


def test_invalid_arguments():
    Xtr, ytr = _data(100, 0)
    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(Xtr, ytr)
    with pytest.raises(ValueError):
        calibrate_confidence(model, Xtr, ytr, alpha=1.5)
    with pytest.raises(ValueError):
        calibrate_confidence(model, Xtr, ytr, mode="bogus")
