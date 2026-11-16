import numpy as np
import pytest

from jspace_oversight.lenses.metrics import top_k_entries


def test_top_k_entries_returns_raw_scores_in_descending_order() -> None:
    indices, scores = top_k_entries(np.array([0.5, -1.0, 3.0, 2.0]), 3)
    assert indices == [2, 3, 0]
    assert scores == [3.0, 2.0, 0.5]


def test_top_k_rejects_matrix_and_invalid_k() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        top_k_entries(np.zeros((2, 2)), 1)
    with pytest.raises(ValueError, match="between"):
        top_k_entries(np.zeros(2), 3)
