"""Contains methods for parent pool selection in one place.

This module brings together methods in which user can select
appropriate parent pool based on individuals fitness score.
"""

import numpy as np

from src.classes.ExperimentConfig import ExperimentConfig


def roulette_selection(fitness_arr: np.ndarray, config: ExperimentConfig) -> list[int]:
    fitness_array = fitness_arr[:, 0].copy()
    fitness_sum = fitness_array.sum()
    if fitness_sum == 0:
        weights = fitness_arr[:, 1].copy()
        biggest_weight = weights.max()
        pseudo_fitness = biggest_weight - weights
        if np.all(pseudo_fitness == 0):
            pseudo_fitness[:] = 1
        fitness_array = pseudo_fitness
        fitness_sum = pseudo_fitness.sum()
    fitness_proportionate = fitness_array / fitness_sum
    proportionate_cfd = np.cumsum(fitness_proportionate.flatten())
    proportionate_cfd[-1] = 1
    r = config.rng.random(config.population_size)
    return np.searchsorted(proportionate_cfd, r).tolist()


def tournament_selection(
    fitness_arr: np.ndarray,
    config: ExperimentConfig,
) -> list[int]:
    tournament_size = 5
    rng = config.rng
    selected_parents = []
    # takes up chunk of population, checks best idx of fitness
    for i in range(config.population_size):
        gladiators = rng.choice(
            config.population_size,
            size=tournament_size,
            replace=False,
        )
        sub = fitness_arr[gladiators]
        order_local = np.lexsort((sub[:, 1], -sub[:, 0]))
        winner_local = order_local[0]
        winner_global = int(gladiators[winner_local])

        selected_parents.append(winner_global)
    return selected_parents


def linear_rank_selection(
    fitness_arr: np.ndarray, config: ExperimentConfig
) -> list[int]:
    rng = config.rng
    SP = config.selection_pressure
    sorted_idx = np.lexsort((-fitness_arr[:, 1], fitness_arr[:, 0]))
    ranks = np.zeros(shape=fitness_arr.shape[0], dtype=np.int64)
    n = len(fitness_arr)
    ranks[sorted_idx] = np.arange(1, n + 1)
    fitness_rank = 2 - SP + 2 * (SP - 1) * (ranks - 1) / (n - 1)
    probability_distribution = fitness_rank / n
    probability_distribution = probability_distribution / probability_distribution.sum()
    parent_arr = rng.choice(
        np.arange(n),
        config.population_size,
        replace=True,
        p=probability_distribution,
    )
    return parent_arr
