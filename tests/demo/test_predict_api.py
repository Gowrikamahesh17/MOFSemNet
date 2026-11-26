"""Tests for the Live Demo endpoint POST /api/demo/predict and the new UI page. Skipped when the saved models are not unpacked."""
import os
import sys

import pytest

from src.demo.predictor import DEFAULT_MODELS_DIR, ROOT

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(DEFAULT_MODELS_DIR, "manifest.json")), reason="saved models not unpacked")
BDC = "OC(=O)c1ccc(cc1)C(=O)O"


@pytest.fixture(scope="module")
def client():
    sys.path.insert(0, os.path.join(ROOT, "backend"))
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_chemistry_only_matches_predictor(client):
    from app.api.routes_predict import _predictor
    r = client.post("/api/demo/predict", json={"smiles": BDC, "metal": "Zn", "graph_type": "semantic_chemberta", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "chemistry_only" and body["known_linker"] and body["mofs_with_this_linker"] == 466
    preds = body["predictions"]
    assert len(preds) == 11 and [p["test_r2"] for p in preds] == sorted((p["test_r2"] for p in preds), reverse=True)
    assert all(p["low90"] <= p["pld"] <= p["high90"] for p in preds)
    direct = _predictor().predict(BDC, "Zn", "chemistry_only", graph_type="semantic_chemberta", k=3)
    assert {p["model_id"]: p["pld"] for p in preds} == {p.model_id: p.pld for p in direct.predictions}
    assert body["nearest_mofs"] and {"refcode", "metal", "pld", "similarity"} <= set(body["nearest_mofs"][0])


def test_pore_values_switch_mode_and_pruned_graph(client):
    r = client.post("/api/demo/predict", json={"smiles": BDC, "metal": "Cu", "lcd": 9.2, "lfs": 8.9,
                                               "graph_type": "hybrid_molt5", "k": 5, "pruning": "black_hole_30"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "with_pore"
    graph_models = [p for p in body["predictions"] if p["graph"] not in ("none", "")]
    assert graph_models and all(p["graph"] == "hybrid_molt5" and p["k"] == 5 and p["pruning"] == "black_hole_30" for p in graph_models)
    assert body["predictions"][0]["test_r2"] > 0.85


@pytest.mark.parametrize("payload, word", [({"smiles": "not-a-smiles", "metal": "Zn"}, "SMILES"),
                                           ({"smiles": BDC, "metal": "Xx"}, "metal"),
                                           ({"smiles": BDC, "metal": "Zn", "k": 4}, "k in"),
                                           ({"smiles": BDC, "metal": "Zn", "graph_type": "nope"}, "graph_type")])
def test_bad_input_is_422(client, payload, word):
    r = client.post("/api/demo/predict", json=payload)
    assert r.status_code == 422 and word in r.json()["detail"]


def test_landing_page_served_and_wired(client):
    if not os.path.exists(os.path.join(ROOT, "frontend/dist/mofworld.html")):
        pytest.skip("frontend not built")
    html = client.get("/mofworld.html").text
    assert "/api/demo/predict" in html and "MOF world" in html
    assert "Case Study" not in html and "CS2" not in html and "UI MOCKUP" not in html


def test_linker_preview(client):
    r = client.get("/api/demo/linker", params={"smiles": "OC(=O)c1ccc(C(O)=O)cc1"}).json()   # BDC written differently
    assert r["valid"] and r["known"] and r["n_mofs"] == 466 and r["formula"] == "C8H6O4" and r["svg"].startswith("<svg")
    assert "terephthalic" in r["name"]
    new = client.get("/api/demo/linker", params={"smiles": "c1ccc2cc3ccccc3cc2c1C(=O)O"}).json()
    assert new["valid"] and not new["known"] and new["n_mofs"] == 0
    assert client.get("/api/demo/linker", params={"smiles": "c1cc("}).json() == {"valid": False}
