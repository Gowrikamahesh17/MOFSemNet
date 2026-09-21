import numpy as np
import pytest

from src.common.config import RuntimeConfig
import src.language.embeddings as embeddings_module


class _FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, smiles_list, batch_size=32):
        self.calls += 1
        return np.array([[len(s)] for s in smiles_list], dtype=np.float32)


def test_get_embedder_rejects_unknown_model():
    config = RuntimeConfig()
    config.language_model = "unknown"  # bypasses __post_init__ validation on purpose, to test get_embedder's own guard
    with pytest.raises(ValueError, match="Unknown language_model"):
        embeddings_module.get_embedder(config)


def test_cached_embed_writes_and_reuses_cache(tmp_path, monkeypatch):
    config = RuntimeConfig(embeddings_cache_dir=str(tmp_path))
    fake = _FakeEmbedder()
    monkeypatch.setattr(embeddings_module, "get_embedder", lambda cfg: fake)

    smiles = ["C", "CCO", "c1ccccc1"]
    first = embeddings_module.cached_embed(smiles, config)
    assert fake.calls == 1
    assert first.shape == (3, 1)

    second = embeddings_module.cached_embed(smiles, config)
    assert fake.calls == 1  # cache hit, embedder not called again
    assert np.array_equal(first, second)


def test_cached_embed_recomputes_on_different_smiles(tmp_path, monkeypatch):
    config = RuntimeConfig(embeddings_cache_dir=str(tmp_path))
    fake = _FakeEmbedder()
    monkeypatch.setattr(embeddings_module, "get_embedder", lambda cfg: fake)

    embeddings_module.cached_embed(["C", "CC"], config)
    embeddings_module.cached_embed(["CCC", "CCCC"], config)
    assert fake.calls == 2
