import numpy as np
import pandas as pd
import pytest

from src.graphs.hybrid_graph import build_hybrid_graph

SAMPLE_SMILES = pd.Series(["c1ccccc1", "c1ccccc1C", "CCO", "CCCCCCCC", "c1ccc(cc1)O"])


def test_hybrid_graph_node_count():
    rng = np.random.default_rng(3)
    embeddings = rng.normal(size=(len(SAMPLE_SMILES), 6))
    graph = build_hybrid_graph(SAMPLE_SMILES, embeddings, k=2, beta=0.5)
    assert graph.number_of_nodes() == len(SAMPLE_SMILES)


def test_hybrid_graph_rejects_invalid_beta():
    rng = np.random.default_rng(3)
    embeddings = rng.normal(size=(len(SAMPLE_SMILES), 6))
    with pytest.raises(ValueError, match="beta"):
        build_hybrid_graph(SAMPLE_SMILES, embeddings, k=2, beta=1.5)


def test_beta_zero_matches_pure_fingerprint_signal():
    from src.graphs.fingerprint_graph import build_fingerprint_graph

    rng = np.random.default_rng(3)
    embeddings = rng.normal(size=(len(SAMPLE_SMILES), 6))
    hybrid = build_hybrid_graph(SAMPLE_SMILES, embeddings, k=2, beta=0.0)
    fingerprint = build_fingerprint_graph(SAMPLE_SMILES, k=2)
    assert set(hybrid.edges()) == set(fingerprint.edges()) or set(hybrid.edges()) == {(v, u) for u, v in fingerprint.edges()}
