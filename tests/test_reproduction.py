"""Tests for parent pairing and mutation helpers in Reproduction."""

import numpy as np
from src.classes.Reproduction import Reproduction


def test_pair_parents_covers_all_indices(
    experiment_config_factory, test_only_pathresolver
) -> None:
    parent_pool = np.arange(6)
    config = experiment_config_factory(
        population_size=6,
        generations=2,
        max_weight=10,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.5,
        mutation_probability=0.0,
        penalty_multiplier=1.0,
    )
    repro = Reproduction(parent_pool, config, test_only_pathresolver)
    flattened = repro.parent_pairs.flatten()
    assert sorted(flattened.tolist()) == sorted(parent_pool.tolist())
    assert repro.parent_pairs.shape == (3, 2)


def test_mutation_flips_bits_when_probability_one(
    experiment_config_factory, test_only_pathresolver
) -> None:
    parent_pool = np.arange(4)
    config = experiment_config_factory(
        population_size=4,
        generations=1,
        max_weight=10,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.5,
        mutation_probability=1.0,
        penalty_multiplier=1.0,
    )
    repro = Reproduction(parent_pool, config, test_only_pathresolver)
    repro.mutation_probability = 1.0
    c1 = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    c2 = np.array([[1, 1], [0, 0]], dtype=np.uint8)

    mutated_c1, mutated_c2 = repro._mutation(c1.copy(), c2.copy())

    np.testing.assert_array_equal(mutated_c1, 1 - c1)
    np.testing.assert_array_equal(mutated_c2, 1 - c2)
