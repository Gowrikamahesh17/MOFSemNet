import numpy as np
import pytest

from src.explainability.embedding_ablation import ablate_embeddings


def _embeddings():
    rng = np.random.default_rng(0)
    unique = rng.normal(size=(5, 8)).astype(np.float32)
    return unique[[0, 0, 1, 2, 2, 2, 3, 4]]  # MOFs 0/1 share a linker, 3/4/5 share a linker


def test_none_returns_an_identical_copy():
    e = _embeddings()
    out = ablate_embeddings(e, "none")
    assert np.array_equal(out, e) and out is not e


@pytest.mark.parametrize("kind", ["shuffle_rows", "random_replacement"])
def test_linker_sharing_is_preserved_but_content_changes(kind):
    e = _embeddings()
    out = ablate_embeddings(e, kind, seed=1)
    assert out.shape == e.shape
    assert np.array_equal(out[0], out[1]) and np.array_equal(out[3], out[4]) and np.array_equal(out[4], out[5])
    assert not np.array_equal(out[0], e[0]) or not np.array_equal(out[2], e[2])


def test_shuffle_rows_keeps_the_same_set_of_vectors():
    e = _embeddings()
    out = ablate_embeddings(e, "shuffle_rows", seed=3)
    assert {tuple(r) for r in out} == {tuple(r) for r in e}


def test_truncate_and_errors():
    e = _embeddings()
    assert ablate_embeddings(e, "truncate_dims", keep_dims=3).shape == (8, 3)
    with pytest.raises(ValueError):
        ablate_embeddings(e, "truncate_dims", keep_dims=99)
    with pytest.raises(ValueError):
        ablate_embeddings(e, "shuffle_dims")
