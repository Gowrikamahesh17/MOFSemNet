#!/usr/bin/env python
"""Step 1 — reproduce the baseline (ported from the prior Case Study, src/baseline/).

Thin CLI wrapper: all actual logic lives in src/baseline/*.py, unmodified from the port except
for import paths. Run from the repo root:
    python scripts/run_step1_baseline.py
"""

from src.baseline.data_ingestion import load_dataset
from src.baseline.feature_engineering import build_features
from src.baseline.graph_construction import build_similarity_graphs, compute_topology_metrics, select_best_graph
from src.common.logging_setup import get_logger

logger = get_logger(__name__)


def main() -> None:
    df = load_dataset("large")
    logger.info(f"Loaded large dataset: {df.shape}")

    features, metadata = build_features(df, scheme="fingerprint")
    logger.info(f"Built features: {features.shape}, metadata keys: {list(metadata.keys())}")

    graphs = build_similarity_graphs(df, dataset="large")
    topology = compute_topology_metrics(graphs)
    best_graph_name = select_best_graph(topology)
    logger.info(f"Selected best graph: {best_graph_name}")
    print(topology)


if __name__ == "__main__":
    main()
