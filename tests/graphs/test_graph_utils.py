import numpy as np

from src.graphs.graph_utils import build_knn_graph_from_similarity, cosine_similarity_block, normalize_embeddings


def test_normalize_embeddings_unit_norm():
    embeddings = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
    normalized = normalize_embeddings(embeddings)
    assert np.allclose(np.linalg.norm(normalized[0]), 1.0)
    assert np.allclose(np.linalg.norm(normalized[1]), 1.0)
    assert np.allclose(normalized[2], [0.0, 0.0])  # zero vector left as zero, not divided by zero


def test_build_knn_graph_basic_properties():
    n = 20
    rng = np.random.default_rng(0)
    embeddings = normalize_embeddings(rng.normal(size=(n, 8)))

    def similarity_fn(row_slice):
        return cosine_similarity_block(embeddings, row_slice)

    graph = build_knn_graph_from_similarity(similarity_fn, n=n, k=3, label="test")

    assert graph.number_of_nodes() == n
    assert graph.number_of_edges() > 0
    for _, _, data in graph.edges(data=True):
        assert -1.0 <= data["weight"] <= 1.0 + 1e-6


def test_build_knn_graph_k_zero_returns_isolated_nodes():
    n = 5
    embeddings = np.eye(5, 3)

    def similarity_fn(row_slice):
        return cosine_similarity_block(embeddings, row_slice)

    graph = build_knn_graph_from_similarity(similarity_fn, n=n, k=0, label="test")
    assert graph.number_of_nodes() == n
    assert graph.number_of_edges() == 0
