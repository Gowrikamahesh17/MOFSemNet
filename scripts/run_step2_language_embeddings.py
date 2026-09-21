#!/usr/bin/env python
"""Step 2 — create language representations: pretrained ChemBERTa/MolT5 embeddings per linker
SMILES, plus a separate one-hot metal representation. Caches embeddings to disk
(config.embeddings_cache_dir) so Step 3 doesn't re-embed on every run.

Usage:
    python scripts/run_step2_language_embeddings.py --model chemberta
    python scripts/run_step2_language_embeddings.py --model molt5
"""

import argparse

from src.baseline.data_ingestion import load_dataset
from src.common.config import LANGUAGE_MODEL_CHOICES, RuntimeConfig
from src.common.logging_setup import get_logger
from src.language.embeddings import cached_embed
from src.language.metal_representation import compute_metal_one_hot

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=LANGUAGE_MODEL_CHOICES, default="chemberta")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    config = RuntimeConfig(language_model=args.model)
    df = load_dataset("large")

    embeddings = cached_embed(df["linker_smiles"].tolist(), config, force_recompute=args.force_recompute)
    logger.info(f"{args.model} embeddings: {embeddings.shape}")

    metal_one_hot, metal_to_index = compute_metal_one_hot(df["metal"])
    logger.info(f"Metal one-hot: {metal_one_hot.shape}, {len(metal_to_index)} unique metals")


if __name__ == "__main__":
    main()
