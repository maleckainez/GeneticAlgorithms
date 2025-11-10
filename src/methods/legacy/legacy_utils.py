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


# DEPRECATED
def single_crossover(
    parent_pairs: np.ndarray[tuple[int, int]],
    rng: np.random.Generator,
    population_file_handle: np.memmap,
    population_file_config: dict[str, any],
    crossover_probability: float = 1,
    mutation_probability: float = 0.1,
) -> None:
    # TODO: docstrings, make the method more modular
    TEMP = find_temp_directory()
    CHILDREN_DAT = TEMP / "children_temp.dat"
    children = np.memmap(
        CHILDREN_DAT,
        dtype=np.uint8,
        mode="w+",
        shape=(
            population_file_config["population_size"],
            population_file_config["genome_length"],
        ),
    )
    for i in range(len(parent_pairs)):
        index_paren1, index_parent2 = parent_pairs[i]
        p1 = population_file_handle[index_paren1]
        p2 = population_file_handle[index_parent2]
        if rng.random() <= crossover_probability:
            cut_place = rng.integers(0, population_file_config["genome_length"])
            children[2 * i] = mutation(
                child=np.concatenate((p1[:cut_place], p2[cut_place:])),
                mutation_probability=mutation_probability,
                rng=rng,
                genome_length=population_file_config["genome_length"],
            )
            children[(2 * i) + 1] = mutation(
                child=np.concatenate((p2[:cut_place], p1[cut_place:])),
                mutation_probability=mutation_probability,
                rng=rng,
                genome_length=population_file_config["genome_length"],
            )
        else:
            children[(2 * i)] = mutation(
                child=p1,
                mutation_probability=mutation_probability,
                rng=rng,
                genome_length=population_file_config["genome_length"],
            )
            children[(2 * i) + 1] = mutation(
                child=p2,
                mutation_probability=mutation_probability,
                rng=rng,
                genome_length=population_file_config["genome_length"],
            )
        children.flush()


# DEPRECATED
def fitness_proportionate_selection(
    fitness_score: np.ndarray,
    rng: np.random.Generator,
    population_file_config: dict[str, any],
) -> list[int]:
    # TODO: docstring
    fitness_sum = 0
    fitness_proportionate = np.ndarray(
        shape=(population_file_config["population_size"], 1), dtype=np.float64
    )
    for i in range(population_file_config["population_size"]):
        fitness_sum += fitness_score[i][0]
    if fitness_sum == 0:
        for i in range(population_file_config["population_size"]):
            fitness_sum += 1
            fitness_score[i][0] = 1
    for i in range(population_file_config["population_size"]):
        fitness_proportionate[i] = fitness_score[i][0] / fitness_sum
    proportionate_cfd = np.cumsum(fitness_proportionate.flatten())
    proportionate_cfd[-1] = 1
    r = rng.random(population_file_config["population_size"])
    return np.searchsorted(proportionate_cfd, r).tolist()


def create_children_temp(population_file_config: dict[str, any]):
    TEMP = find_temp_directory()
    CHILDREN_DAT = TEMP / "children_temp.dat"
    children = np.memmap(
        CHILDREN_DAT,
        dtype=np.uint8,
        mode="w+",
        shape=(
            population_file_config["population_size"],
            population_file_config["genome_length"],
        ),
    )
    return children


def clean_children_temp(
    population_file_config: dict[str, any],
):
    TEMP = find_temp_directory()
    CHILDREN_DAT = TEMP / "children_temp.dat"
    os.remove(population_file_config["filename"])
    os.rename(CHILDREN_DAT, population_file_config["filename"])
