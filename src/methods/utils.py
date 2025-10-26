# --> IMPORTS <--
import os.path
from pathlib import Path
import numpy as np


def load_data(path: str | Path) -> dict:
    """
    This module takes path to the file that contains data in format:
    <value> <weight>
    each line represents different item
    :param path: path to the text file containing data
    :type path: (str | pathlib.Path)
    :return: dictionary {key: [value, weight]}
    :rtype dict[int: list[int,int]]
    :raises FileNotFoundError: if the file does not exist under the provided path
    :raises ValueError: if file is empty, contains letters, have missing data, or is wrongly formatted
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found {path}")
    with open(path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        f.close()
    if not lines:
        raise ValueError("File is empty")
    items = {}
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"Invalid values on line {i+1}: expected 2 values, got {len(parts)}"
            )
        try:
            value, weight = map(int, parts)
        except ValueError:
            raise ValueError(
                f"Invalid values on line {i+1}: received non numeric input {line}"
            )
        items[i] = [value, weight]
    return items


#   DEPRECATED
def create_population(population_size: int, genome_length: int) -> np.ndarray:
    """
    This function replaces the deprecated `create_population()` for improved
    scalability and memory efficiency with large genome sizes.

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


def create_population_file(
    population_size: int, genome_length: int, stream_batch: int, SEED: int
):
    """
    Generates a binary population for a Genetic Algorithm in sequential batches
    and writes it directly to a memory-mapped file ("population.dat") to avoid
    storing the entire population in RAM.

    Each batch is a 2D NumPy array of shape (batch_size, genome_length),
    where batch_size ≤ stream_batch. Data are written to the file incrementally.

    :param population_size: total number of individuals in the population
    :type population_size: int
    :param genome_length: number of genes per individual
    :type genome_length: int
    :param stream_batch: number of individuals generated per write iteration
    :type stream_batch: int
    :param SEED: seed for deterministic random number generation
    :type SEED: int
    :return: memory-mapped array referencing the generated population
    :rtype: numpy.memmap
    """
    population = np.memmap(
        "population.dat",
        dtype=np.uint8,
        mode="w+",
        shape=(population_size, genome_length),
    )
    rng = np.random.default_rng(SEED)
    for start in range(0, population_size, stream_batch):
        stop = min(start + stream_batch, population_size)
        batch = rng.integers(0, 2, size=(stop - start, genome_length), dtype=np.uint8)
        population[start:stop] = batch
        population.flush()
    return population
