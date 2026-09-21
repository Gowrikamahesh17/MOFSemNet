import networkx as nx
import numpy as np
import pytest

from src.sparsification_baselines.k_center_sampling import k_center_sparsify


def _make_graph_and_features(n=60, dim=5, seed=0):
    graph = nx.erdos_renyi_graph(n, 0.1, seed=seed)
    for u, v in graph.edges():
        graph[u][v]["weight"] = 1.0
    features = np.random.default_rng(seed).normal(size=(n, dim))
    return graph, features


def test_k_center_retention_fraction():
    graph, features = _make_graph_and_features(100)
    result = k_center_sparsify(graph, features, retention=0.3, seed=0)
    assert result["graph"].number_of_nodes() == 30


def test_k_center_keeps_fixed_test_nodes():
    graph, features = _make_graph_and_features(100)
    fixed = {5, 6, 7}
    result = k_center_sparsify(graph, features, retention=0.2, seed=0, fixed_test_nodes=fixed)
    assert fixed.issubset(set(result["graph"].nodes()))


def test_k_center_rejects_mismatched_features():
    graph, _ = _make_graph_and_features(20)
    bad_features = np.zeros((5, 3))
    with pytest.raises(ValueError, match="row-aligned"):
        k_center_sparsify(graph, bad_features, retention=0.3, seed=0)


def test_k_center_selects_diverse_points():
    # Two tight clusters far apart: k-center with retention covering both should pick from both.
    cluster_a = np.random.default_rng(1).normal(loc=0, scale=0.01, size=(10, 2))
    cluster_b = np.random.default_rng(2).normal(loc=100, scale=0.01, size=(10, 2))
    features = np.vstack([cluster_a, cluster_b])
    graph = nx.Graph()
    graph.add_nodes_from(range(20))

    result = k_center_sparsify(graph, features, retention=0.2, seed=0)
    kept = set(result["graph"].nodes())
    assert kept & set(range(10)) and kept & set(range(10, 20))
