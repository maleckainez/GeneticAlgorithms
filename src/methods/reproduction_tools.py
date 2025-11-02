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
    parent_pairs: np.ndarray[int],
    seed: int = 2137,
    cross_propab: float = 1,
    mutation_probab: float = 0.1,
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
            children[2 * i] = mutation(
                child=np.concatenate((p1[:cut_place], p2[cut_place:])),
                mutation_probab=mutation_probab,
                rng=rng,
                genome_length=config["genome_length"],
            )
            children[(2 * i) + 1] = mutation(
                child=np.concatenate((p2[:cut_place], p1[cut_place:])),
                mutation_probab=mutation_probab,
                rng=rng,
                genome_length=config["genome_length"],
            )
        else:
            children[(2 * i)] = mutation(
                child=p1,
                mutation_probab=mutation_probab,
                rng=rng,
                genome_length=config["genome_length"],
            )
            children[(2 * i) + 1] = mutation(
                child=p2,
                mutation_probab=mutation_probab,
                rng=rng,
                genome_length=config["genome_length"],
            )
        children.flush()

    os.remove("population.dat")
    os.rename("children_temp.dat", "population.dat")


def mutation(child, rng, genome_length: int, mutation_probab: float = 0.1):
    mask = rng.random(genome_length) < mutation_probab
    child[mask] = 1 - child[mask]
    return child
