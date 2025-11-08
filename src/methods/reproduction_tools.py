import os

import numpy
import numpy as np

from src.methods.utils import find_temp_directory


def parent_pairing(
    parent_pool: numpy.ndarray[int], rng: np.random.Generator
) -> np.ndarray[tuple[int, int]]:
    # TODO: docstrings
    return rng.permutation(parent_pool).reshape(-1, 2)


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


def single_crossover_batched(
    parent_pairs: np.ndarray[tuple[int, int]],
    rng: np.random.Generator,
    population_file_handle: np.memmap,
    population_file_config: dict[str, any],
    crossover_probability: float = 1,
    mutation_probability: float = 0.1,
    batch: int = 500,
):
    children = create_children_temp(population_file_config)
    for start in range(0, len(parent_pairs), batch):
        stop = min(start + batch, len(parent_pairs))
        parent_indices = parent_pairs[start:stop]
        parents1_indx = parent_indices[:, 0]
        parents2_indx = parent_indices[:, 1]

        parents1 = population_file_handle[parents1_indx]
        parents2 = population_file_handle[parents2_indx]

        children1 = parents1.copy()
        children2 = parents2.copy()

        crossover_mask = rng.random(size=stop - start) < crossover_probability
        crossover_points = rng.integers(1, children.shape[1], size=len(parent_indices))

        for i in range(len(parent_indices)):
            if crossover_mask[i] == 1:
                cut_point = crossover_points[i]
                children1[i, cut_point:] = parents2[i, cut_point:]
                children2[i, cut_point:] = parents1[i, cut_point:]
            if mutation_probability > 0:
                mutation_mask = (
                    rng.random(size=children.shape[1]) < mutation_probability
                )
                children1[i, mutation_mask] = 1 - children1[i, mutation_mask]
                children2[i, mutation_mask] = 1 - children2[i, mutation_mask]

        children[start*2:stop*2] = np.concatenate((children1, children2), axis=0)
        children.flush()
    clean_children_temp(population_file_config)


def mutation(
    child,
    genome_length: int,
    rng: np.random.Generator,
    mutation_probability: float = 0.1,
):
    # TODO: docstrings
    child = np.array(child, copy=True)
    if mutation_probability <= 0:
        return child
    mask = rng.random(genome_length) < mutation_probability
    child[mask] = 1 - child[mask]
    return child


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
