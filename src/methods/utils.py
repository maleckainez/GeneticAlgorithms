# --> IMPORTS <--
import os.path
from pathlib import Path
import numpy as np
import json


def load_data(path: str | Path) -> dict:
    """
    This function takes path to the file that contains data in format:
    <value> <weight>
    each line represents different item
    :param path: path to the text file containing data
    :type path: (str | pathlib.Path)
    :return: dictionary {key: [value, weight]}
    :rtype dict[int, list[int,int]]
    :raises FileNotFoundError: if the file does not exist under the provided path
    :raises ValueError: if file is empty, contains letters, have missing data, or is wrongly formatted
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found {path}")
    with open(path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
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
    This function is deprecated!

    Generates an initial binary population for Genetic Algorithm.
    Each individual is a binary genome of length `genomeLength`.

    Gene value 1 means the item is taken, 0 means it is not.
     :param population_size: number of the individuals in the population (height of the numpy matrix)
     :type population_size: int
     :param genome_length: number of genes per individual (must equal number of items)
     :type genome_length: int
     :return: 2D array of shape (population_size, genome_length) with binary values in {0,1}
     :rtype numpy.ndarray
    """
    return np.random.randint(2, size=(population_size, genome_length))


def create_population_file(
    population_size: int, genome_length: int, stream_batch: int, SEED: int
) -> None:
    """
    Generates a binary population for a Genetic Algorithm in sequential batches
    and writes it directly to a memory-mapped file ("population.dat") to avoid
    storing the entire population in RAM.

    Also generates json file to make population more readable and accessible.

    Each batch is a 2D NumPy array of shape (batch_size, genome_length),
    where batch_size ≤ stream_batch (logical requirement). Data are written to the file incrementally.

    :param population_size: total number of individuals in the population
    :type population_size: int
    :param genome_length: number of genes per individual
    :type genome_length: int
    :param stream_batch: number of individuals generated per write iteration
    :type stream_batch: int
    :param SEED: seed for deterministic random number generation
    :type SEED: int
    :return None
    :rtype None
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
    create_memmap_config_json("population", np.uint8, population_size, genome_length)


def create_memmap_config_json(
    filename: str, datatype: type, population_size: int, genome_length
) -> None:
    """
    This function produces config json file for numpy.memmap.
    (This memmap data files are saved as raw binary files with no information about size, or data types)

    :param filename: filename without ".dat" extension
    :type filename: str
    :param datatype: numpy datatype used in the memmap file (it has to be converted onto string)
    :type datatype: numpy.dtype
    :param population_size: number of lines in the file (x dimension)
    :type population_size: int
    :param genome_length: number of bytes in one line (y dimension)
    :type genome_length: int
    :return: None
    """
    config = {
        "filename": filename + ".dat",
        "data_type": np.dtype(datatype).name,
        "population_size": population_size,
        "genome_length": genome_length,
        "filesize": population_size * genome_length,  # file weight in bytes
    }
    with open(filename + ".json", "w") as file:
        json.dump(config, file, indent=4)
