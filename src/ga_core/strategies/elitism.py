"""Encoding-agnostic elitism strategies."""

import numpy as np


def select_elites(
    fitness_arr: np.ndarray,
    elite_percentage: float,
) -> np.ndarray:
    """Select elite individuals based on fitness ranking.

    Args:
        fitness_arr: Array of shape (population_size, 2) with fitness and weight.
        elite_percentage: Fraction of population to preserve as elites (0.0-1.0).

    Returns:
        Array of indices representing elite individuals, sorted by fitness.
    """
    elite_count = int(elite_percentage * len(fitness_arr))
    sorted_idx = np.lexsort((fitness_arr[:, 1], -fitness_arr[:, 0]))
    return sorted_idx[:elite_count]
