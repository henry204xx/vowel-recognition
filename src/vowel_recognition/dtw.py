"""Dynamic time warping implemented with NumPy."""
from __future__ import annotations
import numpy as np


def dtw(sequence_a: np.ndarray, sequence_b: np.ndarray, normalize: str = "length",
        return_path: bool = False):
    """Compute Euclidean-frame DTW cost; length uses alignment-path normalization."""
    a, b = np.asarray(sequence_a, float), np.asarray(sequence_b, float)
    if a.ndim != 2 or b.ndim != 2 or not len(a) or not len(b) or a.shape[1] != b.shape[1]:
        raise ValueError("DTW needs non-empty 2D sequences with equal feature dimensions")
    cumulative = np.full((len(a) + 1, len(b) + 1), np.inf)
    cumulative[0, 0] = 0.
    back = np.zeros((len(a), len(b)), dtype=np.int8)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            candidates = (cumulative[i - 1, j], cumulative[i, j - 1], cumulative[i - 1, j - 1])
            step = int(np.argmin(candidates))
            back[i - 1, j - 1] = step
            cumulative[i, j] = np.linalg.norm(a[i - 1] - b[j - 1]) + candidates[step]
    path: list[tuple[int, int]] = []
    i, j = len(a) - 1, len(b) - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        step = back[i, j]
        if step == 0: i -= 1
        elif step == 1: j -= 1
        else: i, j = i - 1, j - 1
    path.reverse()
    cost = cumulative[-1, -1]
    if normalize == "length": cost /= len(path)
    elif normalize != "raw": raise ValueError("normalize must be 'raw' or 'length'")
    return (float(cost), path) if return_path else float(cost)
