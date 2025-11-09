import numpy as np


def fitness_proportionate_selection(
    fitness_score: np.ndarray,
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
    r = rng.random(population_file_config["population_size"])
    return np.searchsorted(proportionate_cfd, r).tolist()


def tournament_selection(
    fitness_score: np.ndarray,
    rng: np.random.Generator,
    population_file_config: dict[str, any],
    tournament_size: int = 5,
) -> list[int]:

    fitness_list = fitness_score[:, 0]
    selected_parents = []
    # takes up chunk of population, checks best idx of fitness
    for i in range(population_file_config["population_size"]):
        gladiators = rng.choice(
            population_file_config["population_size"],
            size=tournament_size,
            replace=False,
        )
        winner = int(np.argmax(fitness_list[gladiators]))
        selected_parents.append(int(gladiators[winner]))

    return selected_parents


def truncation_selection():
    raise NotImplementedError()
