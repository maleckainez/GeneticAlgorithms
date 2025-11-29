"""Storage utilities for population and children memmap files.

The helpers here encapsulate file naming, validation, and atomic swapping of
population arrays.
"""

import json
import os
import time
from logging import Logger, LoggerAdapter
from pathlib import Path
from typing import Literal, Union

import numpy as np

from .layout import StorageLayout

LoggerType = Union[Logger, LoggerAdapter, None]


def population_filepath(layout: StorageLayout, file_name: str) -> Path:
    """Return memmap path for the main population file.

    The file is stored in the temporary directory associated with the given
    storage layout. The ``.dat`` extension is appended to ``file_name``.

    Args:
        layout: Storage layout that provides the temporary directory.
        file_name: Base name of the population file without extension.

    Returns:
        Path: Absolute path to the population memmap file in the temporary
        directory.
    """
    return layout.temp / f"{file_name}.dat"


def children_filepath(layout: StorageLayout, file_name: str) -> Path:
    """Return memmap path for the temporary children file.

    The file is stored in the temporary directory associated with the given
    storage layout. The ``.dat`` extension is appended to ``file_name`` and
    the name is prefixed with ``child_``.

    Args:
        layout: Storage layout that provides the temporary directory.
        file_name: Base name of the children file without extension.

    Returns:
        Path: Absolute path to the children memmap file in the temporary
        directory.
    """
    return layout.temp / f"child_{file_name}.dat"


def commit_children(
    layout: StorageLayout,
    file_name: str,
    expected_size: int,
    retries: int = 3,
    logger: LoggerType = None,
) -> None:
    """Replace the old population memmap file with the new children file.

    The function validates the size of the children file, then atomically
    replaces the population file using ``os.replace``. Simple retry logic is
    used to reduce the risk of transient ``PermissionError`` exceptions.

    Args:
        layout: Storage layout that provides the temporary directory.
        file_name: Base name used for population and children memmap files
            (without extension).
        expected_size: Required file size of the children file in bytes. A
            mismatch aborts the replacement.
        retries: Maximum number of attempts to perform the replacement before
            failing. A short pause is inserted between attempts.
        logger: Optional logger or logger adapter used for debug or error
            messages describing validation failures and retry attempts.

    Raises:
        RuntimeError: If the children file is missing, has an unexpected
            size, or the replacement fails after all retries.
    """
    child = children_filepath(layout=layout, file_name=file_name)
    population = population_filepath(layout=layout, file_name=file_name)
    if not child.exists():
        if logger is not None:
            logger.error("Missing children file %s", child)
        raise RuntimeError(f"Missing children file {child}")

    if child.stat().st_size != expected_size:
        if logger is not None:
            logger.error(
                "Children size mismatch: %d =/= %d", child.stat().st_size, expected_size
            )
        raise RuntimeError(
            f"Children size mismatch: {child.stat().st_size} =/= {expected_size}"
        )

    last_error = None
    for attempt in range(retries):
        try:
            os.replace(child, population)
            if logger is not None:
                logger.debug("Children commited successfully.")
            return
        except PermissionError as err:
            last_error = err
            if logger is not None:
                logger.warning(
                    "PermissionError while committing children %s -> %s, "
                    "retrying (%d/%d)",
                    child,
                    population,
                    attempt + 1,
                    retries,
                )
            time.sleep(0.2)
    if logger is not None:
        logger.error(
            "Commit failed after %d tries\n" "Dst: %s\n" "Src: %s\n" "With error: %r",
            retries,
            population,
            child,
            last_error,
        )
    raise RuntimeError(
        f"Commit failed after {retries} tries.\n"
        f"Dst: {population}\n"
        f"Src: {child}\n"
        f"With error: {last_error}"
    )


def create_population_file(
    population_size: int,
    genome_length: int,
    temp: Path,
    filename: str,
    data_type: type = np.uint8,
) -> None:
    """Create an empty population memmap file and its JSON config.

    The function allocates a zero-initialised ``np.memmap`` array of shape
    ``(population_size, genome_length)`` in the temporary directory and writes
    a matching JSON configuration file describing its layout and data type.

    Args:
        population_size (int): Number of individuals in the population.
        genome_length (int): Number of genes in a single genome.
        temp (Path): Directory where memmap and JSON files are stored.
        filename (str): Base name for the ``.dat`` and ``.json`` files.
        data_type (type): NumPy-compatible data type used to store the
            population.
    """
    population_dat = temp / f"{filename}.dat"
    population_json = temp / f"{filename}.json"
    np.memmap(
        filename=population_dat,
        dtype=data_type,
        mode="w+",
        shape=(population_size, genome_length),
    )
    create_memmap_config_json(
        population_json, population_dat, data_type, population_size, genome_length
    )


def create_memmap_config_json(
    json_fname_path: Path,
    dat_fname_path: Path,
    datatype: type,
    population_size: int,
    genome_length: int,
) -> None:
    """Create and save JSON config for a memory-mapped array.

    Args:
        json_fname_path (Path): Destination path for the JSON config file.
        dat_fname_path (Path): Path to the corresponding memmap data file.
        datatype (type): NumPy-compatible data type of the memmap.
        population_size (int): Number of rows in the memmap.
        genome_length (int): Number of columns in the memmap.
    """
    genome_length = int(genome_length)
    population_size = int(population_size)
    config = {
        "filename": str(dat_fname_path),
        "data_type": np.dtype(datatype).name,
        "population_size": population_size,
        "genome_length": genome_length,
        "filesize": population_size * genome_length * np.dtype(datatype).itemsize,
    }
    with open(json_fname_path, "w") as file:
        json.dump(config, file, indent=4)


def load_memmap(
    temp: Path,
    filename: str,
    open_mode: Literal[
        "readonly", "r", "copyonwrite", "c", "readwrite", "r+", "write", "w+"
    ] = "r",
) -> tuple[np.memmap, dict]:
    """Load a population memmap and its JSON configuration from disk.

    The JSON configuration is validated against the ``.dat`` file size before
    the memmap is opened.

    Args:
        temp (Path): Directory where memmap and JSON files are stored.
        filename (str): Base name for the memmap and JSON files.
        open_mode (str): NumPy memmap open mode, for example ``"r"`` or
            ``"r+"``.

    Raises:
        FileNotFoundError: If the JSON or data file does not exist.
        ValueError: If the JSON is empty, cannot be parsed, or the size
            recorded in the configuration does not match the data file.

    Returns:
        tuple[np.memmap, dict]: Loaded memmap array and its configuration
            dictionary.
    """
    config_path = temp / (filename + ".json")

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
