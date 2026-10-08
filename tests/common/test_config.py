import pytest

from src.common.config import RuntimeConfig


def test_default_config_is_valid():
    config = RuntimeConfig()
    assert config.dataset_path.endswith("MOFCSD.csv")
    assert config.target_column == "pld_value"
    assert len(config.seeds) >= 5


def test_rejects_unknown_language_model():
    with pytest.raises(ValueError, match="language_model"):
        RuntimeConfig(language_model="gpt4")


def test_rejects_too_few_seeds():
    with pytest.raises(ValueError, match="at least 5 seeds"):
        RuntimeConfig(seeds=(0, 1))


def test_rejects_bad_split_fractions():
    with pytest.raises(ValueError, match="sum to 1.0"):
        RuntimeConfig(train_frac=0.5, val_frac=0.3, test_frac=0.3)


def test_rejects_out_of_range_retention_level():
    with pytest.raises(ValueError, match="retention_levels"):
        RuntimeConfig(retention_levels=(0.3, 1.5))


def test_language_model_hf_id_property():
    assert RuntimeConfig(language_model="chemberta").language_model_hf_id == "seyonec/ChemBERTa-zinc-base-v1"
    assert RuntimeConfig(language_model="molt5").language_model_hf_id == "laituan245/molt5-base"


def test_gravity_weights_normalized_sums_to_one():
    config = RuntimeConfig(gravity_degree_weight=0.5, gravity_betweenness_weight=0.3, gravity_edge_weight_sum_weight=0.2)
    assert sum(config.gravity_weights_normalized) == pytest.approx(1.0)
