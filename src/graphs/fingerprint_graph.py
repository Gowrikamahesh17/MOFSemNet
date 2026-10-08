"""Step 3, fingerprint graph: the same Tanimoto-similarity signal as
src/baseline/graph_construction.py, but rebuilt as a plain top-k graph via
src.graphs.graph_utils so all three graph types (fingerprint, semantic, hybrid) share one
construction method (same k, same blocked top-k builder). src/baseline/graph_construction.py
itself is left untouched (it answers a different question — comparing threshold vs. k-NN
branching for the Step 1 topology experiment) and is not reused here directly; this is a fresh,
minimal builder so the three Step 3 graph types are an apples-to-apples comparison of *signal*
(fingerprint vs. semantic vs. hybrid), not of construction method.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from src.baseline.feature_engineering import compute_fingerprint_matrix
from src.graphs.graph_utils import build_knn_graph_from_similarity, safe_matmul

DEFAULT_K = 10


def build_fingerprint_graph(linker_smiles: pd.Series, k: int = DEFAULT_K) -> nx.Graph:
    """Build a top-k Tanimoto-similarity graph over linker fingerprints."""
    fingerprints = compute_fingerprint_matrix(linker_smiles)
    fp_row_sums = fingerprints.sum(axis=1)
    n = len(linker_smiles)

    def similarity_fn(row_slice):
        intersection = safe_matmul(fingerprints[row_slice], fingerprints.T)
        union = fp_row_sums[row_slice][:, None] + fp_row_sums[None, :] - intersection
        union[union == 0] = 1.0
        return intersection / union

    return build_knn_graph_from_similarity(similarity_fn, n=n, k=k, label="fingerprint")
