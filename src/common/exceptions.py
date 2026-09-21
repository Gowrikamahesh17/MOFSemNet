"""Exceptions for real failure boundaries in the Steps 2-6 pipeline: places where the failure
originates outside this codebase's control (a network call, a foreign file, malformed chemistry
input) rather than a state that internal code guarantees can't happen. Nothing here is a stand-in
for defensive validation of internally-produced data.
"""


class ModelDownloadError(RuntimeError):
    """Raised when a pretrained language model (ChemBERTa/MolT5) can't be fetched from the
    Hugging Face Hub or loaded from the local cache — e.g. no network access and no cached
    weights, or a model name that doesn't exist."""


class InvalidSMILESError(ValueError):
    """Raised when a linker SMILES string can't be parsed by RDKit and therefore can't be
    embedded or turned into a fingerprint. Ported baseline code (src/baseline/data_ingestion.py)
    handles its one known bad-SMILES sentinel by substitution; this is for the general case in
    the new language/graph modules where no such fallback exists."""


class MissingEmbeddingError(RuntimeError):
    """Raised when graph construction (src/graphs/) is asked to build a semantic or hybrid graph
    for a MOF that has no cached language-model embedding — e.g. Step 3 was run before Step 2
    finished, or the embeddings cache on disk is stale relative to the current dataset."""
