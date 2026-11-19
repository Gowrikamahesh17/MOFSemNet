#!/usr/bin/env python
"""Step 5 — compare prediction and efficiency: RF/MLP (no graph) vs. GCN/GraphSAGE (each graph
type x each sparsification method x each retention level), >=5 seeds per configuration. Requires
Step 2's embeddings cache and builds Step 3's graphs directly (rather than reading Step 3's
pickles) so this script is a single reproducible entry point for the full sweep.

Usage:
    python scripts/run_step5_comparison.py --model chemberta
"""

import argparse
import os

from src.baseline.data_ingestion import load_dataset
from src.baseline.feature_engineering import build_features
from src.common.config import LANGUAGE_MODEL_CHOICES, RuntimeConfig
from src.common.logging_setup import get_logger
from src.graphs.fingerprint_graph import build_fingerprint_graph
from src.graphs.hybrid_graph import build_hybrid_graph
from src.graphs.semantic_graph import build_semantic_graph
from src.language.embeddings import cached_embed
from src.evaluation.run_comparison import run_full_comparison

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=LANGUAGE_MODEL_CHOICES, default="chemberta")
    args = parser.parse_args()

    config = RuntimeConfig(language_model=args.model)
    df = load_dataset("large")

    features, _metadata = build_features(df, feature_scheme="fingerprint")
    embeddings = cached_embed(df["linker_smiles"].tolist(), config)

    graphs = {
        "fingerprint": build_fingerprint_graph(df["linker_smiles"]),
        "semantic": build_semantic_graph(embeddings),
        "hybrid": build_hybrid_graph(df["linker_smiles"], embeddings),
    }

    table = run_full_comparison(df, features, graphs, config, embeddings)

    os.makedirs(config.results_dir, exist_ok=True)
    out_path = os.path.join(config.results_dir, f"step5_comparison_{args.model}.csv")
    table.to_csv(out_path, index=False)
    logger.info(f"Saved comparison table to {out_path}")
    print(table)


if __name__ == "__main__":
    main()
