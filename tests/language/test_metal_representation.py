import numpy as np
import pandas as pd

from src.language.metal_representation import compute_metal_one_hot


def test_one_hot_shape_and_mapping():
    metals = pd.Series(["Cu", "Zn", "Cu", "Fe"])
    one_hot, mapping = compute_metal_one_hot(metals)

    assert one_hot.shape == (4, 3)
    assert set(mapping.keys()) == {"Cu", "Zn", "Fe"}
    assert (one_hot.sum(axis=1) == 1).all()
    assert one_hot[0, mapping["Cu"]] == 1.0
    assert one_hot[2, mapping["Cu"]] == 1.0


def test_ignores_missing_values():
    metals = pd.Series(["Cu", None, "Zn"])
    one_hot, mapping = compute_metal_one_hot(metals)
    assert one_hot.shape == (3, 2)
    assert one_hot[1].sum() == 0  # the missing-metal row gets no one-hot entry
