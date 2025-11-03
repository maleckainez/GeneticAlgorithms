# --> IMPORTS <--
import numpy as np
from src.methods import utils


# --> FITNESS <--
def calc_fitness_score(
    ITEMS_VALUE_WEIGHT: dict, MAX_WEIGHT: int, PENALTY_PERCENTAGE: float = 1
):
    population, config = utils.load_memmap("population")
    fitness_score = np.ndarray(shape=(config["population_size"], 2), dtype=np.int64)
    for row in range(len(population)):
        weight = 0
        score = 0
        for gene in range(config["genome_length"]):
            score += int(population[row][gene]) * ITEMS_VALUE_WEIGHT[gene][0]
            weight += int(population[row][gene]) * ITEMS_VALUE_WEIGHT[gene][1]
        if weight > MAX_WEIGHT:
            excess = max(0, weight - MAX_WEIGHT)
            score = int(score * (1 - PENALTY_PERCENTAGE) * (excess / weight))
        fitness_score[row] = [score, weight]
    return fitness_score
