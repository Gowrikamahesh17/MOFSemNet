"""Step 6c: calibrated confidence for the final selected model — turning a point PLD prediction
into a calibrated uncertainty estimate (e.g. via quantile regression, MC dropout, or a conformal
prediction wrapper) so the thesis's trustworthiness analysis has an actual measured confidence
number, not a qualitative claim. Stubbed until Step 5 has picked a final best-performing
configuration to calibrate.
"""

from __future__ import annotations

import numpy as np


def calibrate_confidence(model, X_calibration: np.ndarray, y_calibration: np.ndarray, alpha: float = 0.1) -> dict:
    """Returns {"lower": ..., "upper": ..., "coverage": ...} — a calibrated prediction interval
    at confidence level (1 - alpha), fit on a held-out calibration split (X_calibration must be
    disjoint from both the Step 5 training set and the final untouched test set).

    Not yet implemented — depends on Step 5 having identified the final best-performing model
    configuration to calibrate.
    """
    raise NotImplementedError(
        "calibrate_confidence: depends on Step 5 identifying a final best-performing model "
        "configuration first — calibration is only meaningful for the model that will actually "
        "be reported, not every intermediate configuration in the sweep."
    )
