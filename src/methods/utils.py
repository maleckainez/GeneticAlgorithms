# --> IMPORTS <--
import os.path
import shutil
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


def find_temp_directory():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    TEMP = PROJECT_ROOT / "temp"
    TEMP.mkdir(exist_ok=True)
    return TEMP


def create_population_file(
    population_size: int,
    genome_length: int,
    stream_batch: int,
    rng: np.random.Generator,
    probability_of_failure: float | None = None,
) -> None:
    # TODO: docstrings
    TEMP = find_temp_directory()
    POPULATION_DAT = TEMP / "population.dat"
    POPULATION_JSON = TEMP / "population.json"
    population = np.memmap(
        filename=POPULATION_DAT,
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
        POPULATION_JSON, POPULATION_DAT, np.uint8, population_size, genome_length
    )


def create_memmap_config_json(
    path: Path, dat_path: Path, datatype: type, population_size: int, genome_length
) -> None:
    # TODO: docstrings
    config = {
        "filename": str(dat_path),
        "data_type": np.dtype(datatype).name,
        "population_size": population_size,
        "genome_length": genome_length,
        "filesize": population_size * genome_length,
    }
    with open(path, "w") as file:
        json.dump(config, file, indent=4)


def load_memmap(config_filename: str | None = None, open_mode: str = "r"):
    # TODO: docstrings
    TEMP = find_temp_directory()
    if config_filename is None:
        config_filename = "population"

    config_path = TEMP / (config_filename + ".json")

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


def clear_temp_files():
    # TODO: docstrings
    TEMP = find_temp_directory()
    if TEMP.exists():
        shutil.rmtree(TEMP)


def log_output(
    # TODO: Depreciate, use python logging library, avoid opening population in every iteration -> time and memory consuming!
    iteration: int | None = None,
    best_genome_index: int | None = None,
    fitness: int | None = None,
    weight: int | None = None,
    message: str | None = None,
    path: Path = Path(__file__).resolve().parents[2],
):

    with open(path / "output.log", "a+") as output:
        if iteration or best_genome_index or fitness or weight is not None:
            output.writelines(
                f"Iteration {iteration}:\n"
                f"      index:{best_genome_index}\n"
                f"      fitness: {fitness}\n"
                f"      weight: {weight}\n"
            )
        if message is not None:
            output.writelines(f"Additional message: {message}\n")
    if best_genome_index is not None:
        population, _ = load_memmap()
        with open(path / "best_chromosomes.log", "a+") as best_chromosomes:
            best_chromosomes.writelines(
                f"Best chromosome for iteration {iteration} with fitness {fitness}:\n {population[best_genome_index]}\n"
            )
        population._mmap.close()


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
