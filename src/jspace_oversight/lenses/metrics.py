"""Framework-light numeric utilities for vocabulary readouts."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_numpy(values: Any) -> np.ndarray:
    """Convert NumPy arrays or detached CPU tensors into an array."""
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values)


def top_k_entries(scores: Any, k: int) -> tuple[list[int], list[float]]:
    """Return descending top-k indices and raw scores for one vocabulary row."""
    row = as_numpy(scores)
    if row.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if k < 1 or k > row.size:
        raise ValueError(f"k must be between 1 and {row.size}")
    indices = np.argsort(-row, stable=True)[:k]
    return indices.astype(int).tolist(), row[indices].astype(float).tolist()
