import networkx as nx
import pytest

from src.sparsification_baselines.random_sampling import random_sparsify


def _make_graph(n=50):
    graph = nx.erdos_renyi_graph(n, 0.1, seed=0)
    for u, v in graph.edges():
        graph[u][v]["weight"] = 1.0
    return graph


def test_random_sparsify_retention_fraction():
    graph = _make_graph(100)
    result = random_sparsify(graph, retention=0.3, seed=0)
    assert result["graph"].number_of_nodes() == 30
    assert result["metrics"]["nodes_after"] == 30


def test_random_sparsify_keeps_fixed_test_nodes():
    graph = _make_graph(100)
    fixed = {1, 2, 3}
    result = random_sparsify(graph, retention=0.05, seed=0, fixed_test_nodes=fixed)
    kept = set(result["graph"].nodes())
    assert fixed.issubset(kept)


def test_random_sparsify_rejects_bad_retention():
    graph = _make_graph(10)
    with pytest.raises(ValueError, match="retention"):
        random_sparsify(graph, retention=1.5, seed=0)


def test_random_sparsify_is_seed_reproducible():
    graph = _make_graph(50)
    result_a = random_sparsify(graph, retention=0.4, seed=7)
    result_b = random_sparsify(graph, retention=0.4, seed=7)
    assert set(result_a["graph"].nodes()) == set(result_b["graph"].nodes())
