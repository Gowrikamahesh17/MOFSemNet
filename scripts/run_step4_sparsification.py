#!/usr/bin/env python
"""Step 4 — apply Black Hole sparsification (reused from src.baseline), and compare against the
random and k-center sampling baselines, at the two configured retention levels. Requires Step 3's
cached graphs to already exist.

Usage:
    python scripts/run_step4_sparsification.py --model chemberta --graph-type semantic
"""

import argparse
import pickle
import os

from src.baseline.data_ingestion import load_dataset
from src.common.config import LANGUAGE_MODEL_CHOICES, RuntimeConfig
from src.common.logging_setup import get_logger
from src.language.embeddings import cached_embed
from src.evaluation.run_comparison import compute_global_fixed_test_nodes, k_center_feature_space, sparsify_graph

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=LANGUAGE_MODEL_CHOICES, default="chemberta")
    parser.add_argument("--graph-type", choices=("fingerprint", "semantic", "hybrid"), default="semantic")
    args = parser.parse_args()

    config = RuntimeConfig(language_model=args.model)
    df = load_dataset("large")

    graph_path = os.path.join(config.graphs_cache_dir, f"{args.graph_type}_{args.model}.gpickle")
    with open(graph_path, "rb") as fh:
        graph = pickle.load(fh)

    # k-center's feature space is matched to the graph type being sparsified (fingerprint for
    # the fingerprint graph, embeddings for semantic, both for hybrid) — see
    # src.evaluation.run_comparison.k_center_feature_space for why.
    embeddings = cached_embed(df["linker_smiles"].tolist(), config)
    k_center_features = k_center_feature_space(args.graph_type, df["linker_smiles"], embeddings)
    fixed_test_nodes = compute_global_fixed_test_nodes(len(df), config)

    for method in config.sparsification_methods:
        for retention in config.retention_levels:
            for seed in config.seeds[:1]:  # one seed suffices to report retained nodes/edges here
                result = sparsify_graph(
                    graph, df, method, retention, seed, fixed_test_nodes, k_center_features, config
                )
                logger.info(f"{method} @ retention={retention}: {result['metrics']}")


if __name__ == "__main__":
    main()
