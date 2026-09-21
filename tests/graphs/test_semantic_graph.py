import numpy as np

from src.graphs.semantic_graph import build_semantic_graph


def test_semantic_graph_node_count_matches_embeddings():
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(15, 6))
    graph = build_semantic_graph(embeddings, k=3)
    assert graph.number_of_nodes() == 15


def test_semantic_graph_near_duplicate_embeddings_connect():
    base = np.random.default_rng(2).normal(size=(1, 6))
    embeddings = np.vstack([base, base + 1e-4, base * -1])  # 2 near-identical, 1 near-opposite
    graph = build_semantic_graph(embeddings, k=1)
    assert graph.has_edge(0, 1) or graph.has_edge(1, 0)
