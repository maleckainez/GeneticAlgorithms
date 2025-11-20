"""Defines tests for parent selection methods

This module contiains tests for GA specific selection
methods located in 'src.modules.selection_methods'.
They use numpy random generator to choose parent pool
favoritising more fit individuals.
"""

from collections.abc import Callable

import numpy as np
from src.classes.ExperimentConfig import ExperimentConfig
from src.methods.selection_methods import (
    linear_rank_selection,
    roulette_selection,
    tournament_selection,
)


def test_roulette_happy_path(
    experiment_config_factory: Callable[..., ExperimentConfig],
) -> None:
    config = experiment_config_factory(
        population_size=10,
        generations=5,
        max_weight=100,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.8,
        mutation_probability=0.05,
        penalty_multiplier=10.0,
    )
    fitness_arr = np.array(
        [
            [1, 0.1],
            [2, 0.2],
            [3, 0.3],
            [4, 0.4],
            [5, 0.5],
            [6, 0.6],
            [7, 0.7],
            [8, 0.8],
            [9, 0.9],
            [10, 1.0],
        ]
    )
    fit_sum = fitness_arr[:, 0].sum()
    assert fit_sum != 0
    parents = roulette_selection(fitness_arr, config)
    assert len(parents) == config.population_size
    assert all(0 <= p < config.population_size for p in parents)


def test_roulette_null_fintess(
    experiment_config_factory: Callable[..., ExperimentConfig],
) -> None:
    config = experiment_config_factory(
        population_size=10,
        generations=5,
        max_weight=100,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.8,
        mutation_probability=0.05,
        penalty_multiplier=10.0,
    )
    fitness_arr = np.array(
        [
            [0, 0.1],
            [0, 0.2],
            [0, 0.3],
            [0, 0.4],
            [0, 0.5],
            [0, 0.6],
            [0, 0.7],
            [0, 0.8],
            [0, 0.9],
            [0, 1.0],
        ]
    )
    fit_sum = fitness_arr[:, 0].sum()
    assert fit_sum == 0
    parents = roulette_selection(fitness_arr, config)
    assert len(parents) == config.population_size
    assert all(0 <= p < config.population_size for p in parents)


def test_tournament_selection(
    experiment_config_factory: Callable[..., ExperimentConfig],
) -> None:
    config = experiment_config_factory(
        population_size=10,
        generations=5,
        max_weight=100,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.8,
        mutation_probability=0.05,
        penalty_multiplier=10.0,
    )
    fitness_arr = np.array(
        [
            [1, 0.1],
            [2, 0.2],
            [3, 0.3],
            [4, 0.4],
            [5, 0.5],
            [6, 0.6],
            [7, 0.7],
            [8, 0.8],
            [9, 0.9],
            [10, 1.0],
        ]
    )
    fit_sum = fitness_arr[:, 0].sum()
    assert fit_sum != 0
    parents = tournament_selection(fitness_arr, config)
    assert len(parents) == config.population_size
    assert all(0 <= p < config.population_size for p in parents)


def test_linear_rank_happy_path(
    experiment_config_factory: Callable[..., ExperimentConfig],
) -> None:
    config = experiment_config_factory(
        population_size=10,
        generations=5,
        max_weight=100,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.8,
        mutation_probability=0.05,
        penalty_multiplier=10.0,
        selection_pressure=2,
    )
    fitness_arr = np.array(
        [
            [1, 0.1],
            [2, 0.2],
            [3, 0.3],
            [4, 0.4],
            [5, 0.5],
            [6, 0.6],
            [7, 0.7],
            [8, 0.8],
            [9, 0.9],
            [10, 1.0],
        ]
    )
    fit_sum = fitness_arr[:, 0].sum()
    assert fit_sum != 0
    parents = linear_rank_selection(fitness_arr, config)
    assert len(parents) == config.population_size
    assert all(0 <= p < config.population_size for p in parents)
