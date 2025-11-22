"""Contains methods for parent pool selection in one place.

This module brings together methods in which user can select
appropriate parent pool based on individuals fitness score.
"""

import numpy as np

from src.classes.ExperimentConfig import ExperimentConfig


def roulette_selection(fitness_arr: np.ndarray, config: ExperimentConfig) -> list[int]:
    """
    Select parents using roulette-wheel (fitness-proportionate) selection.

    The first column of ``fitness_arr`` is treated as a fitness value. If the
    sum of fitness values is zero, a pseudo-fitness is derived from the second
    column (e.g. cost/weight) so that lower values correspond to higher
    pseudo-fitness. A cumulative distribution is built from the (pseudo-)fitness
    values and sampled using the RNG from the experiment configuration.

    Args:
        fitness_arr (np.ndarray): 2D array of shape (population_size, 2) where
            column 0 stores fitness and column 1 stores weight.
        config (ExperimentConfig): Experiment configuration holding the RNG
            instance and population size.

    Returns:
        list[int]: Indices of selected parents (with replacement), of length
            ``config.population_size``.
    """
    if config.rng is None:
        raise ValueError("Experiment config was not defined!")
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
    """
    Select parents using tournament selection.

    For each parent to be selected, a fixed-size tournament (subset of
    individuals) is sampled without replacement from the population. The winner
    of the tournament is the individual with the best fitness (column 0 of
    ``fitness_arr``). On ties, the auxiliary value in column 1 is used as a
    secondary criterion via lexicographic ordering.

    Args:
        fitness_arr (np.ndarray): 2D array of shape (population_size, 2) where
            column 0 stores fitness and column 1 stores weight.
        config (ExperimentConfig): Experiment configuration holding the RNG
            instance and population size.

    Returns:
        list[int]: Indices of selected parents (with replacement), of length
            ``config.population_size``.
    """
    if config.rng is None:
        raise ValueError("Experiment config was not defined!")
    tournament_size = 5
    rng = config.rng
    selected_parents = []
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
    """
    Select parents using linear rank-based selection.

    Individuals are sorted and assigned ranks; selection probabilities are then
    computed from these ranks using the linear ranking scheme controlled by
    the selection pressure parameter ``selection_pressure`` from the config.
    Higher ranks receive higher selection probability.

    The ranking is obtained via lexicographic sorting of ``fitness_arr``:
    first by the first column, then by the negated second column, which allows
    combining primary and secondary criteria.

    Args:
        fitness_arr (np.ndarray): 2D array of shape (population_size, 2) where
            column 0 stores fitness and column 1 stores weight used for
            lexicographic ranking.
        config (ExperimentConfig): Experiment configuration holding the RNG
            instance, population size, and the linear selection pressure
            parameter ``selection_pressure`` (in range [1.0, 2.0]).

    Returns:
        list[int]: Indices of selected parents (with replacement), of length
            ``config.population_size``.
    """
    if config.rng is None or config.selection_pressure is None:
        raise ValueError("Experiment config was not defined!")
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
    return parent_arr.tolist()
