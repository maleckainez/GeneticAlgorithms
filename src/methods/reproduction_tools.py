from pathlib import Path

import numpy
import numpy as np
import os

from src.methods.utils import find_temp_directory


def parent_pairing(
        parent_pool: numpy.ndarray[int],
        rng: np.random.Generator
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
        shape=(population_file_config["population_size"], population_file_config["genome_length"]),
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
    os.remove(population_file_config["filename"])
    os.rename(CHILDREN_DAT, population_file_config["filename"])


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
