# --> IMPORTS <--
import numpy as np
from methods import utils


# --> FITNESS <--
def calc_fitness_score(ITEMS_VALUE_WEIGHT: dict, MAX_WEIGHT: int):
    population, config = utils.load_memmap("population")
    fitness_score = np.ndarray(shape=(config["population_size"], 2), dtype=np.int64)
    for row in range(len(population)):
        weight = 0
        score = 0
        for gene in range(config["genome_length"]):
            if (
                weight + int(population[row][gene]) * ITEMS_VALUE_WEIGHT[gene][1]
                > MAX_WEIGHT
            ):
                score = 0
                break
            score += int(population[row][gene]) * ITEMS_VALUE_WEIGHT[gene][0]
            weight += int(population[row][gene]) * ITEMS_VALUE_WEIGHT[gene][1]
        fitness_score[row] = [score, weight]
    return fitness_score
