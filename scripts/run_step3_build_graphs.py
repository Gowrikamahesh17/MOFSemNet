#!/usr/bin/env python
"""Step 3 — prepare features and graphs: fingerprint, semantic, and hybrid graphs over the
large dataset's linker SMILES. Requires Step 2's embeddings cache to already exist for the
configured language model (semantic/hybrid graphs read it via src.language.embeddings.cached_embed).

Usage:
    python scripts/run_step3_build_graphs.py --model chemberta
"""

import argparse
import os
import pickle

from src.baseline.data_ingestion import load_dataset
from src.common.config import LANGUAGE_MODEL_CHOICES, RuntimeConfig
from src.common.logging_setup import get_logger
from src.graphs.fingerprint_graph import build_fingerprint_graph
from src.graphs.hybrid_graph import build_hybrid_graph
from src.graphs.semantic_graph import build_semantic_graph
from src.language.embeddings import cached_embed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=LANGUAGE_MODEL_CHOICES, default="chemberta")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    config = RuntimeConfig(language_model=args.model)
    df = load_dataset("large")
    embeddings = cached_embed(df["linker_smiles"].tolist(), config)

    graphs = {
        "fingerprint": build_fingerprint_graph(df["linker_smiles"], k=args.k),
        "semantic": build_semantic_graph(embeddings, k=args.k),
        "hybrid": build_hybrid_graph(df["linker_smiles"], embeddings, k=args.k),
    }

    os.makedirs(config.graphs_cache_dir, exist_ok=True)
    for graph_type, graph in graphs.items():
        logger.info(f"{graph_type}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        path = os.path.join(config.graphs_cache_dir, f"{graph_type}_{args.model}.gpickle")
        with open(path, "wb") as fh:
            pickle.dump(graph, fh)
        logger.info(f"Saved {graph_type} graph to {path}")


if __name__ == "__main__":
    main()
