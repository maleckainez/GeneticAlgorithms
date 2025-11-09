import numpy as np
from pathlib import Path
from shutil import rmtree

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
    project_root = Path(__file__).resolve().parents[2]
    temp = project_root / "temp"
    temp.mkdir(exist_ok=True)
    return temp


def clear_temp_files():
    # TODO: docstrings
    temp = find_temp_directory()
    if temp.exists():
        rmtree(temp)


def create_output_path():
    ROOT = Path(__file__).resolve().parents[2]
    OUTPUT = ROOT / "output"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    return OUTPUT


# --> DEPRECATED FITNESS <--
def calc_fitness_score(
    value_weight_dict: dict,
    max_weight: int,
    population_file_handle: np.memmap,
    population_file_config: dict[str, any],
    penalty: float = 1,
):
    # TODO: docstrings
    fitness_score = np.ndarray(
        shape=(population_file_config["population_size"], 2), dtype=np.int64
    )
    for row in range(len(population_file_handle)):
        weight = 0
        score = 0
        for gene in range(population_file_config["genome_length"]):
            score += int(population_file_handle[row][gene]) * value_weight_dict[gene][0]
            weight += (
                int(population_file_handle[row][gene]) * value_weight_dict[gene][1]
            )
        if weight > max_weight:
            excess = max(0, weight - max_weight)
            score = int(score * (1 - penalty) * (excess / weight))
        fitness_score[row] = [score, weight]
    return fitness_score
