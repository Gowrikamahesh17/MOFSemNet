"""Requires downloading DeepChem/ChemBERTa-zinc-base-v1 from the Hugging Face Hub (or a local
cache), so these are skipped by default rather than run on every `pytest` invocation. Run with
network access and `--run-network` (see conftest.py) to exercise the real model forward pass.
"""

import pytest

from src.language.chemberta import ChemBERTaEmbedder


@pytest.mark.network
def test_chemberta_embeds_known_smiles(run_network):
    embedder = ChemBERTaEmbedder()
    embeddings = embedder.embed(["c1ccccc1", "CCO"])
    assert embeddings.shape == (2, embedder.embedding_dim)
