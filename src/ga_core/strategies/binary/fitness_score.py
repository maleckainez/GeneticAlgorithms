"""Defines method to calculate fitness of individuals in the population.

This method is encoding-specific for binary genomes.
"""

import numpy as np


def fitness_calculation(
    max_weight: int,
    penalty_factor: float,
    population: np.memmap,
    batch_size: int,
    value_arr: np.ndarray,
    weight_arr: np.ndarray,
) -> np.ndarray:
    """Calculate penalized fitness and total weight for each individual.

    Args:
        max_weight (int): Maximum allowed total weight.
        penalty_factor (float): Factor used to penalize overweight individuals.
        population (np.memmap): Binary population matrix (individuals x genes).
        batch_size (int): Batch size used for streaming computation.
        value_arr (np.ndarray): Value of each gene.
        weight_arr (np.ndarray): Weight of each gene.

    Returns:
        np.ndarray: Array of shape (individuals, 2) with [fitness, weight].
    """
    fitness_score = np.zeros(shape=(population.shape[0], 2), dtype=np.int64)
    for start in range(0, population.shape[0], batch_size):
        stop = min(start + batch_size, population.shape[0])
        current_batch = population[start:stop]
        calculated_scores = current_batch @ value_arr
        calculated_weights = current_batch @ weight_arr
        over_limit_mask = calculated_weights > max_weight
        if penalty_factor == 0:
            penalty_value = calculated_scores
        else:
            penalty_value = np.maximum(
                0, (calculated_weights - max_weight) * (penalty_factor)
            )
        penalized_score = np.where(
            over_limit_mask,
            np.maximum(0, (calculated_scores - penalty_value)),
            calculated_scores,
        )

        fitness_score[start:stop] = np.array(
            (penalized_score, calculated_weights)
        ).transpose()
    return fitness_score
