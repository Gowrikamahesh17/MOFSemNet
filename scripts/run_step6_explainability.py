#!/usr/bin/env python
"""Step 6 — analyse trustworthiness: SHAP for the non-graph baseline, embedding ablation for
the language representations, calibrated confidence for the final model.

Not runnable yet: every function in src/explainability/ raises NotImplementedError until Step 5
has produced trained models to explain (see src/explainability/*.py docstrings). This script is
the intended entry point once that's true — kept here now so the Step 6 wiring (imports, CLI
shape) doesn't have to be invented later.
"""

from src.common.logging_setup import get_logger
from src.explainability.confidence_calibration import calibrate_confidence
from src.explainability.embedding_ablation import run_embedding_ablation
from src.explainability.shap_analysis import compute_shap_values

logger = get_logger(__name__)


def main() -> None:
    raise NotImplementedError(
        "Step 6 depends on Step 5 having produced trained models (a fitted non-graph baseline "
        "to run SHAP on, full-embedding results to ablate against, and a selected final model "
        "to calibrate). Run scripts/run_step5_comparison.py first, then fill in this script's "
        "calls to compute_shap_values / run_embedding_ablation / calibrate_confidence."
    )


if __name__ == "__main__":
    main()
