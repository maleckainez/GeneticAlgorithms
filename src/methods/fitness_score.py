# --> IMPORTS <--
import numpy as np


# --> FITNESS <--
def calc_fitness_score(
    value_weight_dict: dict,
    max_weight: int,
    population_file_handle: np.memmap,
    population_file_config: dict[str, any],
    penalty: float = 1,
):
    # TODO: docstrings
    fitness_score = np.ndarray(
        shape=(population_file_config["population_size"], 2), dtype=np.int64
    )
    for row in range(len(population_file_handle)):
        weight = 0
        score = 0
        for gene in range(population_file_config["genome_length"]):
            score += int(population_file_handle[row][gene]) * value_weight_dict[gene][0]
            weight += (
                int(population_file_handle[row][gene]) * value_weight_dict[gene][1]
            )
        if weight > max_weight:
            excess = max(0, weight - max_weight)
            score = int(score * (1 - penalty) * (excess / weight))
        fitness_score[row] = [score, weight]
    return fitness_score
