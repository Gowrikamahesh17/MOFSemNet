"""Backs the live demo on /implemented: embeds a user-submitted SMILES string with the real,
already-verified ChemBERTa model (src/language), then finds its nearest neighbours among the
real cached embeddings for all 14,296 MOFs in the dataset — genuinely live, genuinely the real
model and real data, nothing canned.

The model and the dataset's embeddings are both expensive to load (model: seconds to minutes;
embeddings: a 23MB array) — both are loaded once, lazily, on first request, and reused for
every request after that. Not loaded at import time so a plain `uvicorn app.main:app` without
a GPU/model cache warm doesn't pay that cost just to serve /api/health.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.models.schemas import EmbeddingDemoResponse, EmbeddingNeighbor, MetalOption

# backend/app/services/embedding_service.py -> repo root is 4 levels up; src/ needs to be
# importable the same way the ML pipeline's own scripts/ expect.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class InvalidSMILESError(ValueError):
    """The submitted string isn't a SMILES RDKit can parse — a real boundary error (user
    input), not an internal bug."""


@lru_cache(maxsize=1)
def _get_embedder():
    from src.common.config import RuntimeConfig
    from src.language.embeddings import get_embedder

    config = RuntimeConfig(language_model=settings.demo_language_model)
    return get_embedder(config)


@lru_cache(maxsize=1)
def _get_reference_data():
    """The real dataset's refcode/metal/pld_value plus the real cached embeddings for all
    14,296 MOFs, row-aligned, for the nearest-neighbour lookup."""
    from src.baseline.data_ingestion import load_dataset
    from src.common.config import RuntimeConfig
    from src.graphs.graph_utils import normalize_embeddings
    from src.language.embeddings import cached_embed

    config = RuntimeConfig(language_model=settings.demo_language_model)
    df = load_dataset("large")
    embeddings = cached_embed(df["linker_smiles"].tolist(), config)
    return df, normalize_embeddings(embeddings)


class InvalidMetalError(ValueError):
    """The requested metal filter isn't one of the metals present in the dataset."""


def list_metals() -> list[MetalOption]:
    """Every metal in the dataset with how many MOFs use it, most common first."""
    df, _ = _get_reference_data()
    counts = df["metal"].astype(str).value_counts()
    return [MetalOption(metal=metal, mofs=int(n)) for metal, n in counts.items()]


def _validate_smiles(smiles: str) -> str:
    from rdkit import Chem

    smiles = smiles.strip()
    if not smiles:
        raise InvalidSMILESError("Enter a SMILES string.")
    if Chem.MolFromSmiles(smiles) is None:
        raise InvalidSMILESError(f"'{smiles}' isn't a SMILES string RDKit can parse.")
    return smiles


def embed_and_find_neighbors(
    smiles: str, metal: str | None = None, top_k: int | None = None
) -> EmbeddingDemoResponse:
    smiles = _validate_smiles(smiles)
    top_k = top_k or settings.demo_top_k

    embedder = _get_embedder()
    embedding = embedder.embed([smiles])[0]

    df, reference_embeddings = _get_reference_data()
    query_norm = embedding / max(np.linalg.norm(embedding), 1e-9)
    similarities = reference_embeddings @ query_norm

    # The embedding encodes the linker only (metal is a separate one-hot, see
    # src/language/metal_representation.py), so a metal choice just restricts which real MOFs
    # are eligible as neighbours; it never changes the embedding or the similarity values.
    if metal:
        metal_mask = (df["metal"].astype(str) == metal).to_numpy()
        if not metal_mask.any():
            raise InvalidMetalError(f"'{metal}' isn't a metal in the dataset.")
        similarities = np.where(metal_mask, similarities, -np.inf)

    top_indices = np.argsort(-similarities)[:top_k]
    neighbors = [
        EmbeddingNeighbor(
            refcode=str(df.iloc[idx]["refcode"]),
            linker_smiles=str(df.iloc[idx]["linker_smiles"]),
            metal=str(df.iloc[idx]["metal"]),
            pld_value=float(df.iloc[idx]["pld_value"]),
            cosine_similarity=round(float(similarities[idx]), 4),
        )
        for idx in top_indices
    ]

    return EmbeddingDemoResponse(
        smiles=smiles,
        metal_filter=metal or None,
        model=settings.demo_language_model,
        embedding_dim=int(embedding.shape[0]),
        embedding_norm=round(float(np.linalg.norm(embedding)), 4),
        neighbors=neighbors,
    )
