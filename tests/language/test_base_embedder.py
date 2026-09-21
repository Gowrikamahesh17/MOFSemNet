import numpy as np

from src.language.base_embedder import BaseSMILESEmbedder


class _FakeEmbedder(BaseSMILESEmbedder):
    """Exercises the batching logic in BaseSMILESEmbedder.embed() without loading a real
    pretrained model — the actual ChemBERTa/MolT5 forward passes are covered separately in
    test_chemberta.py / test_molt5.py, which need network access and are skipped otherwise."""

    model_name = "fake"
    embedding_dim = 4

    def _embed_batch(self, smiles_batch: list[str]) -> np.ndarray:
        # Deterministic stand-in: embedding = [len(smiles), 0, 0, 0] repeated per batch item.
        return np.array([[len(s), 0, 0, 0] for s in smiles_batch], dtype=np.float32)


def test_embed_empty_list_returns_correct_shape():
    embedder = _FakeEmbedder()
    result = embedder.embed([])
    assert result.shape == (0, 4)


def test_embed_batches_correctly():
    embedder = _FakeEmbedder()
    smiles = ["C", "CC", "CCC", "CCCC", "CCCCC"]
    result = embedder.embed(smiles, batch_size=2)
    assert result.shape == (5, 4)
    assert list(result[:, 0]) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_embed_preserves_order():
    embedder = _FakeEmbedder()
    smiles = ["CCCCCCCCCC", "C"]
    result = embedder.embed(smiles, batch_size=1)
    assert result[0, 0] == 10.0
    assert result[1, 0] == 1.0
