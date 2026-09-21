"""Requires downloading laituan245/molt5-base from the Hugging Face Hub (or a local cache) —
see tests/language/test_chemberta.py for why this is network-gated rather than run by default.
"""

import pytest

from src.language.molt5 import MolT5Embedder


@pytest.mark.network
def test_molt5_embeds_known_smiles(run_network):
    embedder = MolT5Embedder()
    embeddings = embedder.embed(["c1ccccc1", "CCO"])
    assert embeddings.shape == (2, embedder.embedding_dim)
