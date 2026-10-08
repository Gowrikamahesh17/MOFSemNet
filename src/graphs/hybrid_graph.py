"""Step 3, hybrid graph: blends the fingerprint (Tanimoto) similarity with the semantic
(language-model cosine) similarity into one similarity score before taking top-k, rather than
merging two separately-built graphs' edges — a weighted blend of the two similarity matrices
lets a MOF-pair that's only strongly similar under one signal still surface as an edge if it
clears the blended threshold, instead of requiring it to independently make both graphs' top-k.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from src.baseline.feature_engineering import compute_fingerprint_matrix
from src.graphs.graph_utils import build_knn_graph_from_similarity, normalize_embeddings, safe_matmul

DEFAULT_K = 10
DEFAULT_BETA = 0.5  # weight on semantic similarity; (1 - beta) on fingerprint Tanimoto similarity


def build_hybrid_graph(
    linker_smiles: pd.Series,
    embeddings: np.ndarray,
    k: int = DEFAULT_K,
    beta: float = DEFAULT_BETA,
) -> nx.Graph:
    """Build a top-k graph from beta * cosine(embeddings) + (1 - beta) * Tanimoto(fingerprints).

    `linker_smiles` and `embeddings` must be row-aligned (same MOF order) — embeddings should
    come from src.language.embeddings.cached_embed on this exact SMILES list.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")

    fingerprints = compute_fingerprint_matrix(linker_smiles)
    fp_row_sums = fingerprints.sum(axis=1)
    normalized_embeddings = normalize_embeddings(embeddings)
    n = len(linker_smiles)

    def similarity_fn(row_slice):
        intersection = safe_matmul(fingerprints[row_slice], fingerprints.T)
        union = fp_row_sums[row_slice][:, None] + fp_row_sums[None, :] - intersection
        union[union == 0] = 1.0
        tanimoto_block = intersection / union

        cosine_block = safe_matmul(normalized_embeddings[row_slice], normalized_embeddings.T)

        return beta * cosine_block + (1 - beta) * tanimoto_block

    return build_knn_graph_from_similarity(similarity_fn, n=n, k=k, label="hybrid")
