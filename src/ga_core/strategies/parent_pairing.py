"""Encoding-agnostic methods to pair parents from given parent pool."""

from typing import Sequence

import numpy as np


def random_pairing(rng: np.random.Generator, parent_pool: Sequence[int]) -> np.ndarray:
    """Pair parents randomly by shuffling the parent pool.

    Args:
        rng: Random number generator for shuffling.
        parent_pool: Sequence of parent indices to be paired.

    Returns:
        Array of shape (n_pairs, 2) with paired parent indices.
    """
    return rng.permutation(parent_pool).reshape(-1, 2)


def sequential_pairing(
    rng: np.random.Generator, parent_pool: Sequence[int]
) -> np.ndarray:
    """Pair parents sequentially without shuffling.

    Args:
        rng: Random number generator (unused but kept for API consistency).
        parent_pool: Sequence of parent indices to be paired.

    Returns:
        Array of shape (n_pairs, 2) with paired parent indices.
    """
    return np.array(parent_pool).reshape(-1, 2)


def best_worst_pairing(
    rng: np.random.Generator, fitness_arr: np.ndarray, parent_pool: np.ndarray
) -> np.ndarray:
    """Pair best individuals with worst individuals.

    Args:
        rng: Random number generator (unused but kept for API consistency).
        fitness_arr: Array of shape (population_size, 2) with fitness and weight.
        parent_pool: Array of parent indices to be paired.

    Returns:
        Array of shape (n_pairs, 2) with paired parent indices.
    """
    n = len(parent_pool)
    sorted_indexes = np.lexsort(
        (fitness_arr[parent_pool, 1], -fitness_arr[parent_pool, 0])
    )
    best = sorted_indexes[: n // 2]
    worst = sorted_indexes[n // 2 :]
    pairs = np.column_stack([best, worst[::-1]])
    return pairs
