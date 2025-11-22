"""Defines population handling functions for Genetic Algorithm runs.

This module provides helper functions to create, load, and handle populations.
To avoid excessive RAM usage, populations are stored on disk as ``np.memmap``
arrays.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np


def create_population_file(
    population_size: int,
    genome_length: int,
    stream_batch: int,
    rng: np.random.Generator,
    temp: Path,
    probability_of_failure: float | None = None,
    filename_constant: str | None = None,
) -> None:
    """Create a memory-mapped population file on disk.

    This function initializes a binary-encoded population for a genetic
    algorithm and stores it as an ``np.memmap`` file on disk, accompanied by
    a JSON metadata file. Each individual is represented as a binary genome
    of length ``genome_length``. Genes are sampled from a Bernoulli
    distribution with probability ``probability_of_failure`` of being 1.

    The population is written in batches, controlled by ``stream_batch``,
    to limit peak memory usage.

    Args:
        population_size (int): even integer value defining amount of individuals
            in single population
        genome_length (int): integer value defining amount of genes in genome
            of single individual
        stream_batch (int): integer value defining size of the batch in which
            they are saved to the drive
        rng (np.random.Generator): random number generator for the experiment
        temp (Path): path to the temporary directory in which memmap is saved
        probability_of_failure (float | None, optional): float variable used to modfiy
            bernoulli distribution of probability. Defaults to standard ``0.5``.
        filename_constant (str | None, optional): Unique filename used to name
            experiment files. Defaults to `population`.
    """
    if filename_constant is None:
        filename_constant = "population"
    population_dat = temp / f"{filename_constant}.dat"
    population_json = temp / f"{filename_constant}.json"
    population = np.memmap(
        filename=population_dat,
        dtype=np.uint8,
        mode="w+",
        shape=(population_size, genome_length),
    )
    if probability_of_failure is None:
        probability_of_failure = 0.5
    for start in range(0, population_size, stream_batch):
        stop = min(start + stream_batch, population_size)
        batch = (
            rng.random(size=(stop - start, genome_length)) < probability_of_failure
        ).astype(np.uint8)
        population[start:stop] = batch
        population.flush()
    create_memmap_config_json(
        population_json, population_dat, np.uint8, population_size, genome_length
    )


def create_memmap_config_json(
    path: Path, dat_path: Path, datatype: type, population_size: int, genome_length: int
) -> None:
    """Create and save JSON config for a memory-mapped array.

    Args:
        path (Path): Destination path for the JSON config file.
        dat_path (Path): Path to the corresponding memmap data file.
        datatype (type): NumPy-compatible data type of the memmap.
        population_size (int): Number of rows in the memmap.
        genome_length (int): Number of columns in the memmap.
    """
    genome_length = int(genome_length)
    population_size = int(population_size)
    config = {
        "filename": str(dat_path),
        "data_type": np.dtype(datatype).name,
        "population_size": population_size,
        "genome_length": genome_length,
        "filesize": population_size * genome_length * np.dtype(datatype).itemsize,
    }
    with open(path, "w") as file:
        json.dump(config, file, indent=4)


def load_memmap(
    temp: Path,
    filename_constant: str | None = None,
    open_mode: Literal[
        "readonly", "r", "copyonwrite", "c", "readwrite", "r+", "write", "w+"
    ] = "r",
) -> tuple[np.memmap, dict]:
    """Load a memory-mapped array and its configuration from disk.

    Args:
        temp (Path): Path to the temporary directory in which memmap is saved.
        filename_constant (str | None): Unique filename used to name experiment files.
            Defaults to `population`.
        open_mode: Mode in which the file is opened. One of
            "readonly", "r", "copyonwrite", "c", "readwrite", "r+", "write", "w+".
            Defaults to "r".


    Raises:
        FileNotFoundError: If population or configuration files are missing.
        ValueError: If population or configuration files are corrupted.

    Returns:
        tuple[np.memmap, dict]: Loaded memmap array and its configuration.
    """
    if filename_constant is None:
        filename_constant = "population"

    config_path = temp / (filename_constant + ".json")

    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} does not exist")
    if config_path.stat().st_size == 0:
        raise ValueError(f"{config_path} is corrupted")

    with open(config_path, "r") as conf_file:
        try:
            config = json.load(conf_file)
        except json.JSONDecodeError:
            raise ValueError(f"{config_path} is corrupted")
    dat_path = Path(config["filename"])
    if not dat_path.exists():
        raise FileNotFoundError(f"{dat_path} does not exist")
    if config["filesize"] != dat_path.stat().st_size:
        raise ValueError(f"{dat_path} is corrupted")
    data_file = np.memmap(
        filename=dat_path,
        dtype=config["data_type"],
        mode=open_mode,
        shape=(config["population_size"], config["genome_length"]),
    )
    return data_file, config
