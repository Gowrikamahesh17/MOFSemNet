import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.common.config import RuntimeConfig
from src.evaluation.run_comparison import (
    compute_global_fixed_test_nodes,
    k_center_feature_space,
    restore_test_nodes,
    run_full_comparison,
    sparsify_graph,
)


def _make_synthetic_dataset(n=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "refcode": [f"MOF{i}" for i in range(n)],
        "pld_value": rng.uniform(2.0, 12.0, size=n).astype(np.float32),
        "metal": rng.choice(["Cu", "Zn"], size=n),
    })
    features = rng.normal(size=(n, 8)).astype(np.float32)
    graph = nx.erdos_renyi_graph(n, 0.2, seed=seed)
    for u, v in graph.edges():
        graph[u][v]["weight"] = float(rng.uniform(0.1, 1.0))
    return df, features, graph


def test_compute_global_fixed_test_nodes_is_deterministic():
    config = RuntimeConfig(test_frac=0.2, val_frac=0.15, train_frac=0.65)
    first = compute_global_fixed_test_nodes(40, config)
    second = compute_global_fixed_test_nodes(40, config)
    assert first == second
    assert len(first) == round(0.2 * 40)


def test_restore_test_nodes_adds_back_missing_nodes():
    original = nx.erdos_renyi_graph(10, 0.5, seed=0)
    for u, v in original.edges():
        original[u][v]["weight"] = 1.0
    pruned = original.subgraph([0, 1, 2, 3]).copy()

    restored = restore_test_nodes(pruned, original, fixed_test_nodes={0, 5, 6})
    assert {0, 5, 6}.issubset(set(restored.nodes()))


SAMPLE_SMILES = pd.Series(["c1ccccc1", "CCO", "c1ccccc1C", "CCCCCCCC"])


def test_k_center_feature_space_fingerprint_uses_only_fingerprint():
    embeddings = np.zeros((len(SAMPLE_SMILES), 5))  # deliberately uninformative
    features = k_center_feature_space("fingerprint", SAMPLE_SMILES, embeddings)
    assert features.shape[0] == len(SAMPLE_SMILES)
    assert features.shape[1] == 1024  # Morgan fingerprint width, not the embedding width


def test_k_center_feature_space_semantic_uses_only_embeddings():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(len(SAMPLE_SMILES), 6))
    features = k_center_feature_space("semantic", SAMPLE_SMILES, embeddings)
    assert features.shape == (len(SAMPLE_SMILES), 6)


def test_k_center_feature_space_hybrid_concatenates_both():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(len(SAMPLE_SMILES), 6))
    features = k_center_feature_space("hybrid", SAMPLE_SMILES, embeddings)
    assert features.shape == (len(SAMPLE_SMILES), 1024 + 6)


def test_k_center_feature_space_rejects_unknown_graph_type():
    with pytest.raises(ValueError, match="Unknown graph_type"):
        k_center_feature_space("magic", SAMPLE_SMILES, np.zeros((len(SAMPLE_SMILES), 3)))


def test_sparsify_graph_random_method():
    df, features, graph = _make_synthetic_dataset()
    config = RuntimeConfig()
    result = sparsify_graph(
        graph, df, "random", retention=0.5, seed=0, fixed_test_nodes={0, 1}, k_center_features=features, config=config
    )
    assert result["graph"].number_of_nodes() == round(0.5 * 40)


def test_sparsify_graph_rejects_unknown_method():
    df, features, graph = _make_synthetic_dataset()
    config = RuntimeConfig()
    with pytest.raises(ValueError, match="Unknown sparsification method"):
        sparsify_graph(
            graph, df, "magic", retention=0.5, seed=0, fixed_test_nodes=set(), k_center_features=features, config=config
        )


@pytest.mark.slow
def test_run_full_comparison_minimal_configuration():
    df, features, graph = _make_synthetic_dataset(n=40)
    embeddings = np.random.default_rng(0).normal(size=(40, 6)).astype(np.float32)
    config = RuntimeConfig(
        graph_types=("fingerprint",),
        sparsification_methods=("random",),
        retention_levels=(0.5,),
        gnn_architectures=("gcn",),
        non_graph_baselines=("random_forest",),
        seeds=(0, 1, 2, 3, 4),
        test_frac=0.2,
        val_frac=0.2,
        train_frac=0.6,
    )
    table = run_full_comparison(df, features, {"fingerprint": graph}, config, embeddings)

    assert len(table) == 2  # one GCN row, one Random Forest row
    assert (table["n_seeds"] == 5).all()
    for metric in ("mae_mean", "rmse_mean", "r2_mean"):
        assert table[metric].notna().all()
