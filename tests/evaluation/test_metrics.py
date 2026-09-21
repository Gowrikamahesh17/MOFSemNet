import pytest

from src.evaluation.metrics import aggregate_seed_runs, build_comparison_table


def test_aggregate_seed_runs_mean_std():
    runs = [
        {"mae": 1.0, "rmse": 2.0, "r2": 0.5, "seed": 0},
        {"mae": 3.0, "rmse": 4.0, "r2": 0.7, "seed": 1},
    ]
    result = aggregate_seed_runs(runs)
    assert result["mae_mean"] == pytest.approx(2.0)
    assert result["mae_std"] == pytest.approx(1.0)
    assert result["n_seeds"] == 2


def test_aggregate_seed_runs_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        aggregate_seed_runs([])


def test_build_comparison_table_shape():
    configurations = {
        ("GCN", "semantic", "black_hole", 0.3): [
            {"mae": 1.0, "rmse": 1.5, "r2": 0.6, "seed": 0},
            {"mae": 1.2, "rmse": 1.6, "r2": 0.55, "seed": 1},
        ],
        ("Random Forest", "none", "none", 1.0): [
            {"mae": 0.9, "rmse": 1.1, "r2": 0.65, "seed": 0},
        ],
    }
    table = build_comparison_table(configurations)
    assert len(table) == 2
    assert {"model", "graph_type", "sparsification_method", "retention_level", "mae_mean", "mae_std"}.issubset(table.columns)
