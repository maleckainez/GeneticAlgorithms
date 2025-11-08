from pathlib import Path

import numpy
import numpy as np
import os

from src.methods.utils import load_memmap


def parent_pairing(
    parent_pool: numpy.ndarray[int], rng: np.random.Generator | None = None
) -> np.ndarray[tuple[int, int]]:
    # TODO: docstrings
    if rng is None:
        rng = np.random.default_rng()
    return rng.permutation(parent_pool).reshape(-1, 2)


def single_crossover(
    parent_pairs: np.ndarray[tuple[int, int]],
    rng: np.random.Generator | None = None,
    crossover_probability: float = 1,
    mutation_probability: float = 0.1,
) -> None:
    # TODO: docstrings, make the method more modular
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    TEMP = PROJECT_ROOT / "temp"
    TEMP.mkdir(exist_ok=True)
    CHILDREN_DAT = TEMP / "children_temp.dat"
    if rng is None:
        rng = np.random.default_rng()
    population, config = load_memmap("population")
    children = np.memmap(
        CHILDREN_DAT,
        dtype=np.uint8,
        mode="w+",
        shape=(config["population_size"], config["genome_length"]),
    )
    for i in range(len(parent_pairs)):
        index_paren1, index_parent2 = parent_pairs[i]
        p1 = population[index_paren1]
        p2 = population[index_parent2]
        if rng.random() <= crossover_probability:
            cut_place = rng.integers(0, config["genome_length"])
            children[2 * i] = mutation(
                child=np.concatenate((p1[:cut_place], p2[cut_place:])),
                mutation_probability=mutation_probability,
                rng=rng,
                genome_length=config["genome_length"],
            )
            children[(2 * i) + 1] = mutation(
                child=np.concatenate((p2[:cut_place], p1[cut_place:])),
                mutation_probability=mutation_probability,
                rng=rng,
                genome_length=config["genome_length"],
            )
        else:
            children[(2 * i)] = mutation(
                child=p1,
                mutation_probability=mutation_probability,
                rng=rng,
                genome_length=config["genome_length"],
            )
            children[(2 * i) + 1] = mutation(
                child=p2,
                mutation_probability=mutation_probability,
                rng=rng,
                genome_length=config["genome_length"],
            )
        children.flush()

    population._mmap.close()
    children._mmap.close()
    os.remove(config["filename"])
    os.rename(CHILDREN_DAT, config["filename"])


def mutation(
    child,
    genome_length: int,
    mutation_probability: float = 0.1,
    rng: np.random.Generator | None = None,
):
    # TODO: docstrings
    child = np.array(child, copy=True)
    if mutation_probability <= 0:
        return child
    mask = rng.random(genome_length) < mutation_probability
    child[mask] = 1 - child[mask]
    return child
