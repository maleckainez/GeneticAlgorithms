# --> IMPORTS <--
import os.path
from pathlib import Path
import numpy as np
import json


# --> UTILS <--
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
                f"Invalid values on line {i+1}: received non integer input {line}"
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
    population_size: int,
    genome_length: int,
    stream_batch: int,
    rng: np.random.Generator | None = None,
    q: float | None = None,
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
    :param rng: previously created RNG with predefined seed for deterministic random number generation
    :type rng: np.random.Generator | None
    :return None
    :rtype None
    """
    population = np.memmap(
        "population.dat",
        dtype=np.uint8,
        mode="w+",
        shape=(population_size, genome_length),
    )
    if q is None:
        q = 0.5
    if rng is None:
        rng = np.random.default_rng()
    for start in range(0, population_size, stream_batch):
        stop = min(start + stream_batch, population_size)
        batch = ((rng.random(size=(stop - start, genome_length))<q).astype(np.uint8))
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


def load_memmap(config_filename: str | None = None, open_mode: str = "r"):
    """
    Load a memory-mapped NumPy array and its associated JSON configuration.

    The function reads metadata from the .json file and validates the
    corresponding .dat binary file before creating the memory map.
    It returns both the opened NumPy memmap object and the loaded configuration
    dictionary for downstream use in other functions.

    :param config_filename: Base name of the JSON configuration file
        (without extension). Defaults to "population".
    :type config_filename: str or None
    :param open_mode: File access mode ("r", "r+", or "w+").
        Defaults to "r".
    :type open_mode: str
    :return: Tuple containing the memory-mapped array and configuration dictionary.
    :rtype: tuple[numpy.memmap, dict]
    :raises FileNotFoundError: If the JSON or data file does not exist.
    :raises ValueError: If either file is empty or inconsistent with metadata.
    """
    if config_filename is None:
        config_filename = "population"
    config_path = config_filename + ".json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} does not exist")
    if os.path.getsize(config_path) == 0:
        raise ValueError(f"{config_path} is corrupted")
    with open(config_path, "r") as conf_file:
        config = json.load(conf_file)
    dat_filename = config["filename"]
    if not os.path.exists(dat_filename):
        raise FileNotFoundError(f"{dat_filename} does not exist")
    if config["filesize"] != os.path.getsize(dat_filename):
        raise ValueError("File is corrupted")

    data_file = np.memmap(
        config["filename"],
        dtype=config["data_type"],
        mode=f"{open_mode}",
        shape=(config["population_size"], config["genome_length"]),
    )
    return data_file, config
