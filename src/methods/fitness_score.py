import json
import os.path

import numpy as np

from methods.utils import create_population


def calc_fitness_score(ITEMS_VALUE_WEIGHT: dict, MAX_WEIGHT: int):
    required_files = ["population.dat", "population.json"]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"{file} does not exist")
        if os.path.getsize(file) == 0:
            raise ValueError(f"{file} is corrupted")
    with open("population.json", "r") as conf_file:
        config = json.load(conf_file)
    if config["filesize"] != os.path.getsize(config["filename"]):
        raise ValueError("File is corrupted")

    population = np.memmap(
        config["filename"],
        dtype=config["data_type"],
        mode="r",
        shape=(config["population_size"], config["genome_length"]),
    )
    fitness_score = np.ndarray(shape=(config["population_size"], 2), dtype=np.int64)
    for row in range(len(population)):
        weight = 0
        score = 0
        for gene in range(config["genome_length"]):

            if weight > MAX_WEIGHT:
                score = 0
                break
            score += int(population[row][gene]) * ITEMS_VALUE_WEIGHT[gene][0]
            weight += int(population[row][gene]) * ITEMS_VALUE_WEIGHT[gene][1]
        fitness_score[row] = [score, weight]
    return fitness_score

    return 0
