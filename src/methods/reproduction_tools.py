import os
import numpy as np

from src.classes.ChildrenHandler import ChildrenHandler
from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver
from src.classes.PopulationHandler import PopulationHandler


def parent_pairing(
    parent_pool: np.ndarray[int],
    config: ExperimentConfig,
) -> np.ndarray[tuple[int, int]]:
    # TODO:
    rng = config.rng
    return rng.permutation(parent_pool).reshape(-1, 2)


def single_crossover_batched(
    parent_pairs: np.ndarray,
    config: ExperimentConfig,
    paths: PathResolver,
    pop_manager: PopulationHandler,
):
    rng = config.rng
    batch = config.stream_batch_size
    crossover_probability = config.crossover_probability
    mutation_probability = config.mutation_probability
    population = pop_manager.get_pop_handle()
    children_manager = ChildrenHandler(
        config=config, paths=paths, genome_length=population.shape[1]
    )
    children = children_manager.get_children_handle()

    for start in range(0, len(parent_pairs), batch):
        stop = min(start + batch, len(parent_pairs))
        parent_indices = parent_pairs[start:stop]
        parents1_indx = parent_indices[:, 0]
        parents2_indx = parent_indices[:, 1]

        parents1 = population[parents1_indx]
        parents2 = population[parents2_indx]

        children1 = parents1.copy()
        children2 = parents2.copy()

        crossover_mask = rng.random(size=stop - start) < crossover_probability
        first_crossover_points = rng.integers(
            1, children1.shape[1] - 1, size=len(parent_indices)
        )
        second_crossover_points = rng.integers(
            first_crossover_points + 1, children1.shape[1], size=len(parent_indices)
        )

        crossover_points = np.column_stack(
            (first_crossover_points, second_crossover_points)
        )

        for i in range(len(parent_indices)):
            if crossover_mask[i] == 1:
                first_cut_point, second_cut_point = crossover_points[i]
                children1[i, first_cut_point:] = parents2[i, first_cut_point:]
                children1[i, second_cut_point:] = parents1[i, second_cut_point:]
                children2[i, first_cut_point:] = parents1[i, first_cut_point:]
                children2[i, second_cut_point:] = parents2[i, second_cut_point:]
            if mutation_probability > 0:
                mutation_mask = (
                    rng.random(size=children.shape[1]) < mutation_probability
                )
                children1[i, mutation_mask] = 1 - children1[i, mutation_mask]
                children2[i, mutation_mask] = 1 - children2[i, mutation_mask]

        children[start * 2 : stop * 2] = np.concatenate((children1, children2), axis=0)
        children.flush()
    children_manager.close()


def double_crossover_batched(
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

        children[start * 2 : stop * 2] = np.concatenate((children1, children2), axis=0)
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
