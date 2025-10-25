# --> IMPORTS <--
import os.path
from pathlib import Path
import numpy as np

# -----------SEED-------------
np.random.seed(2137)
# ----------------------------


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


def create_population_stream(
    population_size: int, genome_length: int, stream_batch: int
):
    """
    Lazily generates batches of binary genomes for a Genetic Algorithm population.

    Each yielded batch is a 2D NumPy array of shape (batch_size, genome_length),
    where batch_size ≤ stream_batch. This approach prevents allocating the entire
    population in memory at once.

    :param population_size: total number of individuals to generate
    :type population_size: int
    :param genome_length: number of genes per individual (usually number of items)
    :type genome_length: int
    :param stream_batch: number of individuals generated per iteration
    :type stream_batch: int
    :yield: NumPy array representing a batch of individuals (binary genomes)
    :rtype: numpy.ndarray[uint8]
    """
    for pos in range(0, population_size, stream_batch):
        yield np.random.randint(
            2,
            size=(min(stream_batch, population_size - pos), genome_length),
            dtype=np.uint8,
        )
