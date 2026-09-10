"""Minimal, transparent spatial diagnostics without future information."""

from __future__ import annotations

import numpy as np

from .features import haversine_distance_matrix


def knn_weights(latitude: np.ndarray, longitude: np.ndarray, k: int = 8) -> np.ndarray:
    distances = haversine_distance_matrix(latitude, longitude)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    weights = np.zeros_like(distances)
    weights[np.arange(len(weights))[:, None], neighbors] = 1.0
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def morans_i(values: np.ndarray, weights: np.ndarray) -> float:
    centered = values.astype(float) - np.mean(values)
    denominator = np.dot(centered, centered)
    if denominator == 0:
        return np.nan
    return float(len(values) / weights.sum() * (centered @ weights @ centered) / denominator)


def morans_i_permutation_test(
    values: np.ndarray, weights: np.ndarray, permutations: int = 999, seed: int = 42
) -> dict[str, float]:
    observed = morans_i(values, weights)
    rng = np.random.default_rng(seed)
    simulated = np.array([morans_i(rng.permutation(values), weights) for _ in range(permutations)])
    p_value = (np.sum(np.abs(simulated) >= abs(observed)) + 1) / (permutations + 1)
    return {
        "morans_i": observed,
        "permutation_p_value_two_sided": float(p_value),
        "permutations": permutations,
    }
