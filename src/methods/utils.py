# --> IMPORTS <--
import os.path

import numpy as np


def load_data(path: str) -> dict:
    """
    This module takes path to the file that contains data in format:
    <value> <weight>
    each line represents different item
    :param path: path to the text file containing data
    :type path: str
    :return: dictionary {key: [value] [weight]}
    :rtype dict[int: list[int,int]]
    :raises FileNotFoundError: if the dile does not exist under the provided path
    :raises ValueError: if file contains invalid of blank lines
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found {path}")
    with open(path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        f.close()
    if not lines:
        raise ValueError("File is empty or corrupted")
    items = {}
    for i in range(len(lines)):
        items[i] = [int(x) for x in lines[i].split(" ")]
    return items


def create_population(population_size: int, genome_length: int) -> np.ndarray:
    """
    Generates an initial binary population for Genetic Algorithm.
    Each individual is a binary genome of length `genomeLength`.

    Gene value 1 means the item is taken, 0 means it is not.
     :param population_size: number of the individuals in the population (height of the numpy matrix)
     :type population_size: int
     :param genome_length: number of genes per individual (must equal number of items)
     :type genome_length: int
     :return: 2D array of shape (populationSize, genomeLength) with binary values in {0,1}
     :rtype numpy ndarray of ints
    """
    return np.random.randint(2, size=(population_size, genome_length))
