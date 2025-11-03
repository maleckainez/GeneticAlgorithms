# --> IMPORTS <--
import numpy as np
from src.methods import utils


# --> FITNESS <--
def calc_fitness_score(value_weight_dict: dict, max_weight: int, penalty: float = 1):
    population, config = utils.load_memmap("population")
    fitness_score = np.ndarray(shape=(config["population_size"], 2), dtype=np.int64)
    for row in range(len(population)):
        weight = 0
        score = 0
        for gene in range(config["genome_length"]):
            score += int(population[row][gene]) * value_weight_dict[gene][0]
            weight += int(population[row][gene]) * value_weight_dict[gene][1]
        if weight > max_weight:
            excess = max(0, weight - max_weight)
            score = int(score * (1 - penalty) * (excess / weight))
        fitness_score[row] = [score, weight]
    return fitness_score
