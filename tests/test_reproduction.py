"""Defines tests for reproduction class."""

import numpy as np
from src.classes.ChildrenHandler import ChildrenHandler
from src.classes.Reproduction import Reproduction


def _create_pop_handler(dummy_pop_manager, temp_file):
    population = np.memmap(
        filename=f"{temp_file}.dat", dtype=np.uint8, mode="w+", shape=(10, 5)
    )
    population[:] = [
        [0, 0, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
    ]
    return dummy_pop_manager(population)


def test_reproduction_single_crossover_no_mutation(
    experiment_config_factory, test_only_pathresolver, dummy_pop_manager, temp_file
):
    config = experiment_config_factory(
        population_size=10,
        generations=1,
        max_weight=100,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.9,
        mutation_probability=0,
        penalty_multiplier=0,
    )
    parent_pool = np.array([1, 5, 6, 9, 2, 0, 0, 7, 6, 6], dtype=np.uint8)
    pop_manager = _create_pop_handler(dummy_pop_manager, temp_file)
    reproduction = Reproduction(
        parent_pool=parent_pool, config=config, paths=test_only_pathresolver
    )
    handler = ChildrenHandler(
        config=config,
        paths=test_only_pathresolver,
        genome_length=5,
    )
    reproduction.single_crossover(pop_manager, handler)


def test_reproduction_double_crossover_no_mutation(
    experiment_config_factory, test_only_pathresolver, dummy_pop_manager, temp_file
):
    config = experiment_config_factory(
        population_size=10,
        generations=1,
        max_weight=100,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.9,
        mutation_probability=0,
        penalty_multiplier=0,
    )
    parent_pool = np.array([1, 5, 6, 9, 2, 0, 0, 7, 6, 6], dtype=np.uint8)
    pop_manager = _create_pop_handler(dummy_pop_manager, temp_file)
    reproduction = Reproduction(
        parent_pool=parent_pool, config=config, paths=test_only_pathresolver
    )
    handler = ChildrenHandler(
        config=config,
        paths=test_only_pathresolver,
        genome_length=5,
    )
    reproduction.double_crossover(pop_manager, handler)


def test_reproduction_single_crossover_with_mutation(
    experiment_config_factory, test_only_pathresolver, dummy_pop_manager, temp_file
):
    config = experiment_config_factory(
        population_size=10,
        generations=1,
        max_weight=100,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.9,
        mutation_probability=0.1,
        penalty_multiplier=0,
    )
    parent_pool = np.array([1, 5, 6, 9, 2, 0, 0, 7, 6, 6], dtype=np.uint8)
    pop_manager = _create_pop_handler(dummy_pop_manager, temp_file)
    reproduction = Reproduction(
        parent_pool=parent_pool, config=config, paths=test_only_pathresolver
    )
    handler = ChildrenHandler(
        config=config,
        paths=test_only_pathresolver,
        genome_length=5,
    )
    reproduction.single_crossover(pop_manager, handler)
