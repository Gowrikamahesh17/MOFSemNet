"""Shared blocked top-k similarity-graph builder for src/graphs/semantic_graph.py and
src/graphs/hybrid_graph.py.

Same blocking strategy as src/baseline/graph_construction.py::build_similarity_graphs (never
materialize the full N x N similarity matrix — process row blocks) since N=14,296 nodes makes a
dense matrix ~1.6GB per float64 copy, and this project needs to stay runnable on a laptop, not
just a university server.
"""

from __future__ import annotations

import time

import networkx as nx
import numpy as np

from src.common.logging_setup import get_logger

logger = get_logger(__name__)

BLOCK_SIZE = 1500


def build_knn_graph_from_similarity(
    similarity_fn,
    n: int,
    k: int,
    label: str = "",
    block_size: int = BLOCK_SIZE,
) -> nx.Graph:
    """Build an undirected top-k similarity graph without ever forming the full N x N matrix.

    `similarity_fn(row_slice)` must return a (len(row_slice), n) similarity block for the given
    row indices; self-similarity is masked internally so callers don't need to.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    max_k = min(k, max(0, n - 1))
    if max_k == 0:
        return graph

    start = time.time()
    for block_start in range(0, n, block_size):
        block_end = min(block_start + block_size, n)
        row_slice = slice(block_start, block_end)

        sim_block = similarity_fn(row_slice)

        local_rows = np.arange(block_end - block_start)
        global_rows = local_rows + block_start
        sim_block[local_rows, global_rows] = -1.0  # mask self-similarity

        top_idx = np.argpartition(-sim_block, max_k - 1, axis=1)[:, :max_k]
        for local_r in range(block_end - block_start):
            gi = local_r + block_start
            candidates = top_idx[local_r]
            sims = sim_block[local_r, candidates]
            for c, s in zip(candidates, sims):
                if s > -1.0:  # skip the masked self-similarity slot on tiny graphs
                    graph.add_edge(gi, int(c), weight=float(s))

        logger.info(f"[{label}] top-{k} similarity block {block_end}/{n} rows done")

    logger.info(f"[{label}] graph construction took {time.time() - start:.1f}s for {n} nodes, k={k}")
    return graph


def safe_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`a @ b`, with numpy's overflow/invalid-value/divide-by-zero RuntimeWarnings suppressed.

    On macOS, numpy's float32 matmul runs through Apple's Accelerate BLAS backend, which has a
    documented bug (numpy/numpy#21798, #21826) where it raises spurious overflow/invalid-value
    FP exception flags on some matmuls even when every input and output value is finite and
    correct — verified here by comparing against np.einsum (bit-identical result, no NaN/Inf)
    on this project's actual fingerprint/embedding matrices. Scoped to just this call rather
    than a broader warnings filter, so a genuine NaN/Inf bug elsewhere still surfaces normally.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return a @ b


def cosine_similarity_block(embeddings: np.ndarray, row_slice: slice) -> np.ndarray:
    """Cosine similarity between embeddings[row_slice] and all of embeddings. Embeddings are
    L2-normalized once by the caller (see normalize_embeddings) so this reduces to a matmul."""
    return safe_matmul(embeddings[row_slice], embeddings.T)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms
