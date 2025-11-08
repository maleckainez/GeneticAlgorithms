import numpy as np


def fitness_proportionate_selection(
    fitness_score: np.ndarray,
    parent_group_size: int,
    rng: np.random.Generator,
    population_file_config: dict[str, any],
) -> list[int]:
    # TODO: docstring
    fitness_sum = 0
    fitness_proportionate = np.ndarray(
        shape=(population_file_config["population_size"], 1), dtype=np.float64
    )
    for i in range(population_file_config["population_size"]):
        fitness_sum += fitness_score[i][0]
    if fitness_sum == 0:
        for i in range(population_file_config["population_size"]):
            fitness_sum += 1
            fitness_score[i][0] = 1
    for i in range(population_file_config["population_size"]):
        fitness_proportionate[i] = fitness_score[i][0] / fitness_sum
    proportionate_cfd = np.cumsum(fitness_proportionate.flatten())
    proportionate_cfd[-1] = 1
    r = rng.random(parent_group_size)
    return np.searchsorted(proportionate_cfd, r).tolist()


def tournament_selection():
    raise NotImplementedError()


def truncation_selection():
    raise NotImplementedError()
