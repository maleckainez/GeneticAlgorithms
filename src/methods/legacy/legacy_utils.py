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


# FINALLY DEPRECATED
def log_output(
    log_path: Path,
    filename_constant: str | None = None,
    iteration: int | None = None,
    best_genome_index: int | None = None,
    fitness: int | None = None,
    weight: int | None = None,
    message: str | None = None,
    genome: np.ndarray | None = None,
):
    if filename_constant is None:
        filename_constant = ""
    with open(log_path / f"result_{filename_constant}.log", "a+") as output:
        if (
            iteration is not None
            or best_genome_index is not None
            or fitness is not None
            or weight is not None
        ):
            output.writelines(
                f"Iteration {iteration}:\n"
                f"      index:{best_genome_index}\n"
                f"      fitness: {fitness}\n"
                f"      weight: {weight}\n"
            )
        if message is not None:
            output.writelines(f"{message}\n")
    if genome is not None:
        with open(
            log_path / f"chromosomes_{filename_constant}.log", "a+"
        ) as best_chromosomes:
            if fitness > 0:

                genome = "".join(str(i) for i in genome.tolist())
                best_chromosomes.writelines(
                    f"Best chromosome for iteration {iteration} with fitness {fitness}:\n{genome}\n"
                )
            else:
                best_chromosomes.writelines(
                    f"No chromosome for iteration {iteration} with fitness higher than 0\n"
                )
