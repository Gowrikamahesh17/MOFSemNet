"""Tests for the live-demo predictor. Skipped when the saved models are not unpacked in outputs/models/."""
import os

import numpy as np
import pandas as pd
import pytest

from src.demo.predictor import DEFAULT_MODELS_DIR, ROOT

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(DEFAULT_MODELS_DIR, "manifest.json")), reason="saved models not unpacked")
PRED_FILE = os.path.join(ROOT, "outputs/results/predictions_all_models.csv.gz")


@pytest.fixture(scope="module")
def P():
    from src.demo.predictor import MOFPredictor
    return MOFPredictor()


@pytest.fixture(scope="module")
def known():
    if not os.path.exists(PRED_FILE):
        pytest.skip("run scripts/export_model_predictions.py first")
    allp = pd.read_csv(PRED_FILE)
    raw = pd.read_csv(os.path.join(ROOT, "data/raw/MOFCSD.csv"), index_col=0)
    row = allp[allp.split == "test"].iloc[3]
    return row, raw.loc[row.refcode]


def test_known_mof_matches_saved_predictions(P, known):
    """Tree models and no-edge GNNs do not depend on the graph, so a known MOF must reproduce the saved predictions."""
    row, x = known
    r = P.predict(row.linker_smiles, row.metal, "with_pore", lcd=x["Largest Cavity Diameter"], lfs=x["Largest Free Sphere"])
    checked = 0
    for p in r.predictions:
        if p.graph == "none":
            assert abs(p.pld - row[p.model_id]) < 2e-3, p.model_id
            checked += 1
    assert checked == 8  # 5 tree models + 3 no-edge GNNs
    assert r.known_linker and r.predictions[0].test_r2 >= r.predictions[-1].test_r2


def test_new_linker_gives_finite_predictions_with_intervals(P):
    smiles = "OC(=O)c1ccc(cc1)C(=O)O"   # known linker; the graph path is exercised without needing the language model download
    r = P.predict(smiles, "Zn", "chemistry_only", graph_type="fingerprint", k=5, pruning="black_hole_30")
    assert len(r.predictions) == 11  # 5 trees + 3 no-edge GNNs + 3 GNNs on the chosen graph
    for p in r.predictions:
        assert np.isfinite(p.pld) and p.low90 <= p.pld <= p.high90
    assert len(r.nearest_mofs) == 5


@pytest.mark.parametrize("bad", [dict(smiles="not a smiles", metal="Zn"), dict(smiles="OC=O", metal="Xx")])
def test_bad_inputs_are_rejected(P, bad):
    with pytest.raises(ValueError):
        P.predict(bad["smiles"], bad["metal"])


def test_pore_mode_needs_pore_values(P):
    with pytest.raises(ValueError):
        P.predict("OC=O", "Zn", "with_pore")
