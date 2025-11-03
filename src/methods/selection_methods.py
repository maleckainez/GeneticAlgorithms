import numpy as np
from src.methods.utils import load_memmap


def fitness_proportionate_selection(
    fitness_score: np.ndarray, parent_group_size:int, seed:int=2137
) -> np.ndarray[int]:
    fitness_sum = 0
    population, config = load_memmap()
    fitness_propotionate = np.ndarray(
        shape=(config["population_size"], 1), dtype=np.float64
    )
    for i in range(config["population_size"]):
        fitness_sum += fitness_score[i][0]
    if fitness_sum == 0:
        for i in range(config["population_size"]):
            fitness_sum += 1
            fitness_score[i][0] = 1
    for i in range(config["population_size"]):
        fitness_propotionate[i] = fitness_score[i][0] / fitness_sum
    propotionate_cfd = np.cumsum(fitness_propotionate.flatten())
    propotionate_cfd[-1] = 1
    r = np.random.default_rng(seed).random(parent_group_size)
    return np.searchsorted(propotionate_cfd, r).tolist()


def tournament_selection():
    raise NotImplementedError()


def truncation_selection():
    raise NotImplementedError()
