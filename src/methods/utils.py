# --> IMPORTS <--
import logging
import os.path
from pathlib import Path
import numpy as np
import json

import yaml

from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver


# --> UTILS <--
def load_data(path: str | Path) -> np.ndarray:
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
    data_in_lines = []
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
        data_in_lines.append([value, weight])
        items = np.array(data_in_lines, dtype=np.int64)
    return items


def create_population_file(
    population_size: int,
    genome_length: int,
    stream_batch: int,
    rng: np.random.Generator,
    temp: Path,
    probability_of_failure: float | None = None,
    filename_constant: str | None = None,
) -> None:
    # TODO: docstrings
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
    # TODO: docstrings
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
    open_mode: str = "r",
) -> tuple[np.memmap, dict[str, any]]:
    # TODO: docstrings
    if filename_constant is None:
        filename_constant = "population"

    config_path = temp / (filename_constant + ".json")

    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} does not exist")
    if config_path.stat().st_size == 0:
        raise ValueError(f"{config_path} is corrupted")

    with open(config_path, "r") as conf_file:
        config = json.load(conf_file)

    dat_path = Path(config["filename"])
    if not dat_path.exists():
        raise FileNotFoundError(f"{dat_path} does not exist")
    if config["filesize"] != dat_path.stat().st_size:
        raise ValueError("File is corrupted")
    data_file = np.memmap(
        filename=dat_path,
        dtype=config["data_type"],
        mode=open_mode,
        shape=(config["population_size"], config["genome_length"]),
    )
    return data_file, config


def final_screen():
    print(
        r""" 
           ______      __           __      __  _           
          / ____/___ _/ /______  __/ /___ _/ /_(_)___  ____  
         / /   / __ `/ / ___/ / / / / __ `/ __/ / __ \/ __ \ 
        / /___/ /_/ / / /__/ /_/ / / /_/ / /_/ / /_/ / / / / 
        \____/\__,_/_/\___/\__,_/_/\__,_/\__/_/\____/_/ /_/  
            _______       _      __             __   __     
           / ____(_)___  (_)____/ /_  ___  ____/ /  / /     
          / /_  / / __ \/ / ___/ __ \/ _ \/ __  /  / /     
         / __/ / / / / / (__  ) / / /  __/ /_/ /  /_/       
        /_/   /_/_/ /_/_/____/_/ /_/\___/\__,_/  (_)        
                                                            """
    )


def load_yaml_config(filepath: Path | str) -> dict:
    with open(filepath, "r") as file:
        yaml_file = yaml.safe_load(file)

    yaml_config = {
        "data_filename": yaml_file["data"]["filename"],
        "max_weight": yaml_file["data"]["max_weight"],
        "population_size": yaml_file["population"]["size"],
        "generations": yaml_file["population"]["generations"],
        "stream_batch_size": yaml_file["population"]["stream_batch_size"],
        "selection_type": yaml_file["selection"]["type"],
        "crossover_type": yaml_file["genetic_operators"]["crossover_type"],
        "crossover_probability": yaml_file["genetic_operators"][
            "crossover_probability"
        ],
        "mutation_probability": yaml_file["genetic_operators"]["mutation_probability"],
        "penalty": yaml_file["genetic_operators"]["penalty_multiplier"],
        "seed": yaml_file["experiment"]["seed"],
        "experiment_identifier": yaml_file["experiment"]["identifier"],
        "log_level": yaml_file["experiment"]["log_level"],
    }

    return yaml_config
