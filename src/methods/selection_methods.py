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


def linear_rank_selection():
    raise NotImplementedError()
