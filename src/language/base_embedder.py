"""Common interface both pretrained chemical language model adapters implement, so
src/graphs/semantic_graph.py and src/evaluation/ never need to know which model produced an
embedding — they just call `.embed(smiles_list)` and get back real pretrained-model output.

Every embed() implementation must return the model's own forward-pass output (pooled hidden
states) with no fallback to a hand-built or invented descriptor if the model can't be loaded —
that failure must raise src.common.exceptions.ModelDownloadError instead, per the thesis
guardrail that language representations are never a hallucinated stand-in for the real model.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np


class BaseSMILESEmbedder(ABC):
    """Extracts a fixed-size embedding per SMILES string from a pretrained model's own hidden
    states, with on-disk caching keyed by the SMILES string itself."""

    model_name: str  # e.g. "chemberta" — used as the cache subdirectory name
    embedding_dim: int

    @abstractmethod
    def _embed_batch(self, smiles_batch: list[str]) -> np.ndarray:
        """Run the pretrained model forward and return a (batch, embedding_dim) array.
        Subclasses implement this; callers use embed()."""

    def embed(self, smiles_list: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of SMILES strings, batching the forward passes.

        Returns a (len(smiles_list), embedding_dim) float32 array in the same order as the
        input. Does not deduplicate or cache in-memory across calls — see
        src.language.embeddings.cached_embed for the disk-cached entry point used by the
        Step 3 graph-construction pipeline.
        """
        if not smiles_list:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        chunks = []
        for start in range(0, len(smiles_list), batch_size):
            batch = smiles_list[start : start + batch_size]
            chunks.append(self._embed_batch(batch))
        return np.concatenate(chunks, axis=0).astype(np.float32)


def cache_path(cache_dir: str, model_name: str) -> str:
    return os.path.join(cache_dir, f"{model_name}_embeddings.npz")
