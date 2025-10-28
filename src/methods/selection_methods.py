import numpy as np
from methods.utils import load_memmap


def fitness_proportionate_selection(
    fitness_score: np.ndarray, parent_group_size=5, seed=2137
):
    fitness_sum = 0
    population, config = load_memmap()
    fitness_propotionate = np.ndarray(
        shape=(config["population_size"], 1), dtype=np.float64
    )
    for i in range(config["population_size"]):
        fitness_sum += fitness_score[i][0]
    for i in range(config["population_size"]):
        fitness_propotionate[i] = fitness_score[i][0] / fitness_sum
    propotionate_cfd = np.cumulative_sum(fitness_propotionate.flatten())
    propotionate_cfd[-1] = 1

    print(f"Population: \n {population}\n")
    print(f"cfd: \n {propotionate_cfd}\n")
    print("parent group: \n")
    rng = np.random.default_rng(seed)
    for i in range(parent_group_size):
        r = rng.random()
        l = np.searchsorted(propotionate_cfd, r)
        print(f"random value: {r}\n parent: {population[l]}")



def tournament_selection():
    raise NotImplementedError()


def truncation_selection():
    raise NotImplementedError()
