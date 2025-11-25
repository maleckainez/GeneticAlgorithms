"""Defines method to calculate fitness of individuals in the population."""

import numpy as np

from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PopulationHandler import PopulationHandler


def fitness_calculation(
    max_weight: int,
    penalty_factor: float,
    population: np.memmap,
    batch: int,
    value_arr: np.ndarray,
    weight_arr: np.ndarray,
) -> np.ndarray:
    """Calculate penalized fitness and total weight for each individual.

    Args:
        max_weight (int): Maximum allowed total weight.
        penalty_factor (float): Factor used to penalize overweight individuals.
        population (np.memmap): Binary population matrix (individuals x genes).
        batch (int): Batch size used for streaming computation.
        value_arr (np.ndarray): Value of each gene.
        weight_arr (np.ndarray): Weight of each gene.

    Returns:
        np.ndarray: Array of shape (individuals, 2) with [fitness, weight].
    """
    fitness_score = np.zeros(shape=(population.shape[0], 2), dtype=np.int64)
    for start in range(0, population.shape[0], batch):
        stop = min(start + batch, population.shape[0])
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


def fitness_class_adapter(
    value_weight_arr: np.ndarray,
    config: ExperimentConfig,
    pop_manager: PopulationHandler,
) -> np.ndarray:
    """Adapter computing batched fitness from config and population handler.

    Args:
        value_weight_arr (np.ndarray): Array with columns [value, weight] per gene.
        config (ExperimentConfig): Experiment configuration with fitness settings.
        pop_manager (PopulationHandler): Provides memmap handle to the population.

    Returns:
        np.ndarray: Array of shape (individuals, 2) with [fitness, weight].
    """
    max_weight = config.max_weight
    penalty_factor = config.penalty

    population = pop_manager.get_pop_handle()
    batch = config.stream_batch_size
    # Defensive guard: ExperimentConfig.__post_init__ guarantees batch is not None.
    # Marked as no cover because this branch should be unreachable in normal usage.
    if batch is None:  # pragma: no cover
        batch = 500
    value = value_weight_arr[:, 0]
    weight = value_weight_arr[:, 1]
    assert population is not None
    return fitness_calculation(
        max_weight=max_weight,
        penalty_factor=penalty_factor,
        population=population,
        batch=batch,
        value_arr=value,
        weight_arr=weight,
    )


# Legacy alias
calc_fitness_score_batched = fitness_class_adapter
