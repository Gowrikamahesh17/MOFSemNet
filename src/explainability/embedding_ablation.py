"""Step 6b: embedding ablation for the language representations — measuring how much of the
semantic/hybrid graph's advantage (if any) actually comes from the pretrained language model's
embedding dimensions, versus surviving with a degraded/shuffled/truncated embedding. Stubbed
until Step 5 has full-embedding results to ablate against.
"""

from __future__ import annotations

import numpy as np


def run_embedding_ablation(
    embeddings: np.ndarray,
    ablation_type: str,
    config,
) -> dict:
    """Rebuilds the semantic graph (src.graphs.semantic_graph) from an ablated version of
    `embeddings` and re-runs the Step 5 GCN/GraphSAGE comparison on it, to isolate how much of
    the language model's contribution comes from real chemical signal vs. incidental structure.

    ablation_type in {"shuffle_dims", "truncate_dims", "random_replacement"} (exact set TBD
    once Step 5 baselines exist to compare against).

    Not yet implemented — depends on Step 5's full-embedding results as the comparison point.
    """
    raise NotImplementedError(
        "run_embedding_ablation: depends on Step 5's full-embedding GCN/GraphSAGE results "
        "existing first, as the baseline the ablated variant is compared against."
    )
