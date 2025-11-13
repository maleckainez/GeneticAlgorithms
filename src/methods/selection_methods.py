from unittest import findTestCases

import numpy as np

from src.classes.ExperimentConfig import ExperimentConfig


def roulette_selection(fitness_arr: np.ndarray, config: ExperimentConfig) -> list[int]:
    fitness_sum = fitness_arr[:, 0].sum()
    if fitness_sum == 0:
        fitness_arr[:, 0] = 1
        fitness_sum = fitness_arr.shape[0]
    fitness_proportionate = fitness_arr[:, 0] / fitness_sum
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
    fitness_score_arr = fitness_arr[:, 0]
    selected_parents = []
    # takes up chunk of population, checks best idx of fitness
    for i in range(config.population_size):
        gladiators = rng.choice(
            config.population_size,
            size=tournament_size,
            replace=False,
        )
        winner = int(np.argmax(fitness_score_arr[gladiators]))
        selected_parents.append(int(gladiators[winner]))
    return selected_parents


def _tournament_tie_breaker(gladiators: list[int], fitness_arr: np.ndarray):
    raise NotImplementedError()


def linear_rank_selection(fitness_arr: np.ndarray, config: ExperimentConfig):
    rng = config.rng
    SP = 2
    sorted_idx = np.lexsort((-fitness_arr[:, 1], fitness_arr[:, 0]))
    ranks = np.zeros(shape=fitness_arr.shape[0], dtype=np.int64)
    n = len(fitness_arr)
    ranks[sorted_idx] = np.arange(1, n + 1)
    # Consider Nind the number of individuals in the population, Pos the position of an individual in this population
    # (least fit individual has Pos=1, the fittest individual Pos=Nind) and SP the selective pressure.
    # The fitness value for an individual is calculated as:
    # Fitess(pos) = 2 - SP + 2*(SP-1)*((pos-1)/Nind-1)

    fitness_rank = 2 - SP + 2 * (SP - 1) * (ranks - 1) / (n)
    probability_distribution = fitness_rank / n
    # distribution safeguard
    probability_distribution = probability_distribution / probability_distribution.sum()
    parent_arr = rng.choice(
        np.arange(n),
        config.population_size,
        replace=True,
        p=probability_distribution,
    )
    if parent_arr.size % 2 != 0:
        parent_arr = np.append(
            parent_arr,
            rng.choice(
                np.arange(n),
                config.population_size,
                replace=True,
                p=probability_distribution,
            ),
        )
    return parent_arr
