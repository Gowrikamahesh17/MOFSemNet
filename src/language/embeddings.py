"""Factory + disk-caching entry point for Step 2. This is what src/graphs/semantic_graph.py and
scripts/run_step2_language_embeddings.py actually call — they never import ChemBERTaEmbedder or
MolT5Embedder directly, so switching config.language_model is a one-line change (this is the
"switch between them on the UI later" hook the config was built for).
"""

from __future__ import annotations

import os

import numpy as np

from src.common.config import RuntimeConfig
from src.common.logging_setup import get_logger
from src.language.base_embedder import BaseSMILESEmbedder, cache_path

logger = get_logger(__name__)


def get_embedder(config: RuntimeConfig) -> BaseSMILESEmbedder:
    """Construct the embedder named by config.language_model. Import is done lazily inside each
    branch so `import src.language.embeddings` doesn't force-load `transformers` + torch just to
    read the module, and so choosing "chemberta" never requires MolT5's tokenizer to be
    importable (and vice versa)."""
    if config.language_model == "chemberta":
        from src.language.chemberta import ChemBERTaEmbedder

        return ChemBERTaEmbedder(config)
    if config.language_model == "molt5":
        from src.language.molt5 import MolT5Embedder

        return MolT5Embedder(config)
    raise ValueError(f"Unknown language_model {config.language_model!r}")


def cached_embed(smiles_list: list[str], config: RuntimeConfig, force_recompute: bool = False) -> np.ndarray:
    """Embed `smiles_list` with config.language_model, reusing a disk cache keyed by the exact
    list of SMILES + model name. Re-embedding 14,296 MOFs on every graph-construction run would
    dominate runtime, so Step 3+ should always go through this rather than calling an embedder's
    .embed() directly.

    The cache is invalidated (recomputed) whenever the input SMILES list differs from what's
    cached, since a stale cache silently mismatched to a different dataset slice would be a hard
    bug to notice downstream.
    """
    path = cache_path(config.embeddings_cache_dir, config.language_model)

    if not force_recompute and os.path.exists(path):
        cached = np.load(path, allow_pickle=True)
        if list(cached["smiles"]) == list(smiles_list):
            logger.info(f"Loaded cached {config.language_model} embeddings from {path}")
            return cached["embeddings"]
        logger.info(f"Cache at {path} doesn't match requested SMILES list, recomputing.")

    embedder = get_embedder(config)
    embeddings = embedder.embed(smiles_list)

    os.makedirs(config.embeddings_cache_dir, exist_ok=True)
    np.savez_compressed(path, smiles=np.array(smiles_list, dtype=object), embeddings=embeddings)
    logger.info(f"Cached {len(smiles_list)} {config.language_model} embeddings to {path}")

    return embeddings
