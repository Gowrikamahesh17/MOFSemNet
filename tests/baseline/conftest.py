"""Shared fixtures for the ported Step 1 test suite.

These were referenced by the copied tests (small_df, large_df, tiny_graph) but the conftest.py
that originally defined them wasn't part of the files copied into this repo — reconstructed here
from how each fixture is used across tests/baseline/*.py (exact required columns, tiny_graph's
"two 5-cycles joined by one bridge edge" structure documented in
test_black_hole_sparsification.py's test_centrality_is_scoped_to_the_community_not_the_whole_graph).
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

# A pool of real, RDKit-parseable organic SMILES standing in for MOF linkers — enough distinct
# structures (and enough repeats) to give k-NN (k up to 10) and Louvain community detection
# something non-trivial to work with, without needing the real 14,296-row dataset in unit tests.
_LINKER_SMILES_POOL = [
    "c1ccccc1",            # benzene
    "c1ccc(cc1)C(=O)O",    # benzoic acid
    "c1ccc2c(c1)ccc(c2)",  # naphthalene-like
    "c1ccc(cc1)c1ccccc1",  # biphenyl
    "c1ccc(cc1)O",         # phenol
    "c1ccc(cc1)N",         # aniline
    "C1CCCCC1",            # cyclohexane
    "CCCCCCCC",            # octane
    "c1ccc(cc1)C(=O)c1ccccc1",  # benzophenone
    "c1cc(ccc1C(=O)O)C(=O)O",   # terephthalic acid
    "c1ccc2[nH]ccc2c1",    # indole-like
    "CCOC(=O)c1ccccc1",    # ethyl benzoate
]

_METALS = ["Cu", "Zn", "Fe", "Co"]
_CATEGORY_ORDER = ["nonporous", "small pore", "medium pore", "large pore"]


def _canonical_columns(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    linker_smiles = [_LINKER_SMILES_POOL[i % len(_LINKER_SMILES_POOL)] for i in range(n)]
    return {
        "refcode": [f"MOF{i:04d}" for i in range(n)],
        "linker_smiles": linker_smiles,
        "metal": rng.choice(_METALS, size=n).tolist(),
        "pld_category": rng.choice(_CATEGORY_ORDER, size=n).tolist(),
        "pld_value": rng.uniform(2.0, 14.0, size=n).astype(np.float32),
    }


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Canonical-schema DataFrame with the small (2,000-MOF) dataset's extra columns
    (metal_feat_0..5, mof_index) needed by the compact feature scheme."""
    n = 25
    rng = np.random.default_rng(0)
    data = _canonical_columns(n, seed=0)
    for i in range(6):
        data[f"metal_feat_{i}"] = rng.normal(size=n).astype(np.float32)
    data["mof_index"] = list(range(n))
    return pd.DataFrame(data)


@pytest.fixture
def large_df() -> pd.DataFrame:
    """Canonical-schema DataFrame with the large (14,296-MOF) dataset's extra columns
    (pore-geometry) needed by the fingerprint feature scheme."""
    n = 25
    rng = np.random.default_rng(1)
    data = _canonical_columns(n, seed=1)
    data["Largest Cavity Diameter"] = rng.uniform(3.0, 20.0, size=n).astype(np.float32)
    data["Largest Free Sphere"] = rng.uniform(2.0, 15.0, size=n).astype(np.float32)
    return pd.DataFrame(data)


@pytest.fixture
def tiny_graph() -> nx.Graph:
    """10 nodes (0-9): two 5-cycles (0-1-2-3-4-0 and 5-6-7-8-9-5) joined by a single bridge
    edge (4-5) — deliberately weakly connected so Louvain community detection reliably splits
    it into exactly the two cycles, letting community-scoped-vs-whole-graph centrality tests
    tell the difference."""
    graph = nx.Graph()
    cycle_a = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    cycle_b = [(5, 6), (6, 7), (7, 8), (8, 9), (9, 5)]
    for u, v in cycle_a + cycle_b:
        graph.add_edge(u, v, weight=1.0)
    graph.add_edge(4, 5, weight=0.1)  # weak bridge between the two communities
    return graph
