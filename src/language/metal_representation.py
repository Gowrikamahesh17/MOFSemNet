"""Separate metal representation for Step 2.

Metal identity ("Cu", "Zn", ...) isn't a SMILES string, so it doesn't go through a chemical
language model — it gets its own one-hot encoding, kept as a distinct representation from the
linker embedding rather than concatenated inside src/language/embeddings.py. This mirrors how
src/baseline/feature_engineering.py::compute_fingerprint_scheme_features already one-hot-encodes
metal alongside the Morgan fingerprint for the large dataset; this module exists so Steps 3-5 can
reuse the same metal encoding without importing baseline's fingerprint-scheme function (whose
fingerprint half of the output isn't wanted here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metal_one_hot(metals: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    """One-hot encode a metal column. Returns (one_hot array of shape (n, n_unique_metals),
    metal_to_index mapping) — the mapping is returned so it can be persisted alongside cached
    embeddings/graphs and reused consistently across train/val/test splits and future runs."""
    unique_metals = sorted(metals.dropna().unique().tolist())
    metal_to_index = {metal: i for i, metal in enumerate(unique_metals)}

    one_hot = np.zeros((len(metals), len(unique_metals)), dtype=np.float32)
    for row_idx, metal in enumerate(metals.values):
        if metal in metal_to_index:  # missing values (NaN/None) get no one-hot entry
            one_hot[row_idx, metal_to_index[metal]] = 1.0

    return one_hot, metal_to_index
