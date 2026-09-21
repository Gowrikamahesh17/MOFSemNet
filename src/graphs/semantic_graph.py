"""Step 3, semantic graph: a top-k similarity graph built from cosine similarity over Step 2's
language-model embeddings, instead of Tanimoto over fingerprints. This is the graph type the
whole thesis hinges on — testing whether a similarity structure built from a *different* kind of
information than each node's own features gives the GNN a genuine edge over the fingerprint
graph, where the prior Case Study found the graph added nothing because it was built from
information already present in the node features.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.graphs.graph_utils import build_knn_graph_from_similarity, cosine_similarity_block, normalize_embeddings

DEFAULT_K = 10


def build_semantic_graph(embeddings: np.ndarray, k: int = DEFAULT_K) -> nx.Graph:
    """Build a k-NN graph over MOFs, edges weighted by cosine similarity between their linker
    language-model embeddings (src.language.embeddings.cached_embed output). Node indices match
    the row order of `embeddings`, matching src.baseline.graph_construction's convention so
    downstream code (Black Hole sparsification, GNN training) doesn't need a semantic-graph
    special case for node indexing.
    """
    normalized = normalize_embeddings(embeddings)
    n = normalized.shape[0]

    def similarity_fn(row_slice):
        return cosine_similarity_block(normalized, row_slice)

    return build_knn_graph_from_similarity(similarity_fn, n=n, k=k, label="semantic")
