import pytest

from src.explainability.embedding_ablation import run_embedding_ablation


def test_run_embedding_ablation_raises_until_step5_results_exist():
    """Step 6 (embedding ablation) is stubbed until Step 5 has full-embedding results to ablate
    against. This test documents the current contract; replace with real assertions once
    run_embedding_ablation is implemented."""
    with pytest.raises(NotImplementedError):
        run_embedding_ablation(embeddings=None, ablation_type="shuffle_dims", config=None)
