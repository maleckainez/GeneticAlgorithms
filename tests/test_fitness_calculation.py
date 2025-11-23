"""Defines tests for fitness score calculation module.

This module contiains test specific for fitness calculation
method located in 'src/methods/fitness_score'.
"""

import numpy as np
from src.methods.fitness_score import fitness_calculation, fitness_class_adapter


def test_fitness_penalty_factor_zero_discards_overweight_individuals() -> None:
    population = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    value_arr = np.array([5, 5, 5])
    weight_arr = np.array([4, 6, 2])

    result = fitness_calculation(
        max_weight=10,
        penalty_factor=0,
        population=population,
        batch=1,
        value_arr=value_arr,
        weight_arr=weight_arr,
    )

    expected = np.array([[10, 6], [0, 12]])
    np.testing.assert_array_equal(result, expected)


def test_fitness_penalty_reduces_score_and_keeps_weight_intact() -> None:
    population = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.uint8)
    value_arr = np.array([40, 30, 20])
    weight_arr = np.array([20, 40, 100])

    result = fitness_calculation(
        max_weight=50,
        penalty_factor=2.0,
        population=population,
        batch=2,
        value_arr=value_arr,
        weight_arr=weight_arr,
    )

    expected = np.array([[50, 60], [0, 160]])
    np.testing.assert_array_equal(result, expected)


def test_fitness_calculation_consistent_across_batches() -> None:
    population = np.array(
        [
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 1],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    value_arr = np.array([3, 5, 7, 11])
    weight_arr = np.array([2, 4, 6, 8])

    small_batch = fitness_calculation(
        max_weight=10**6,
        penalty_factor=1.0,
        population=population,
        batch=2,
        value_arr=value_arr,
        weight_arr=weight_arr,
    )
    large_batch = fitness_calculation(
        max_weight=10**6,
        penalty_factor=1.0,
        population=population,
        batch=10,
        value_arr=value_arr,
        weight_arr=weight_arr,
    )

    np.testing.assert_array_equal(small_batch, large_batch)


def test_fitness_class_adapter_uses_config_and_population_handle(
    experiment_config_factory,
    dummy_pop_manager,
) -> None:
    population = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    value_weight_arr = np.array([[10, 5], [20, 8], [15, 4]])

    config = experiment_config_factory(
        population_size=2,
        generations=1,
        max_weight=12,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.5,
        mutation_probability=0.1,
        penalty_multiplier=2.0,
        stream_batch=1,
    )

    result = fitness_class_adapter(
        value_weight_arr=value_weight_arr,
        config=config,
        pop_manager=dummy_pop_manager(population),
    )

    expected = np.array([[25, 9], [35, 17]])
    np.testing.assert_array_equal(result, expected)
