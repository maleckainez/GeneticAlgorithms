import numpy
import numpy as np
import os

from src.methods.utils import load_memmap, create_memmap_config_json


def parent_pairing(
    parent_pool: numpy.ndarray[int], seed: int = 2137
) -> np.ndarray[int]:
    rng = np.random.default_rng(seed)
    return rng.permutation(parent_pool).reshape(-1, 2)


def single_crossover(
    parent_pairs: np.ndarray[int], seed: int = 2137, cross_propab: float = 1
) -> None:
    rng = np.random.default_rng(seed)
    population, config = load_memmap("population")
    print(f"Population length: {len(population)}")
    children = np.memmap(
        "children_temp.dat",
        dtype=np.uint8,
        mode="w+",
        shape=(config["population_size"], config["genome_length"]),
    )
    for i in range(len(parent_pairs)):
        indx_p1, indx_p2 = parent_pairs[i]
        p1 = population[indx_p1]
        p2 = population[indx_p2]
        if rng.random() <= cross_propab:
            cut_place = rng.integers(0, config["genome_length"])
            children[2 * i] = np.concatenate((p1[:cut_place], p2[cut_place:]))
            children[(2 * i) + 1] = np.concatenate((p2[:cut_place], p1[cut_place:]))
        else:
            children[(2 * i)] = p1
            children[(2 * i) + 1] = p2
        children.flush()

    os.remove("population.dat")
    os.rename("children_temp.dat", "population.dat")

def mutation():
    raise NotImplementedError
    # TODO: Simple implementation of bit flip
