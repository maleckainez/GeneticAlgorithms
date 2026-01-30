"""Unit tests for binary mutation."""

import numpy as np
import pytest
from src.ga_core.strategies.binary.mutation import bit_flip_mutation


@pytest.fixture
def binary_population():
    """Sample binary population."""
    return np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1]], dtype=np.uint8)


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


def test_bit_flip_mutation_modifies_in_place(binary_population, rng):
    """Mutation modifies the array in place."""
    original_id = id(binary_population)
    result = bit_flip_mutation(binary_population, 0.2, rng)
    assert id(result) == original_id


def test_bit_flip_mutation_flips_bits(rng):
    """Mutation flips 0→1 and 1→0."""
    population = np.array([[0, 0, 0, 0, 0]], dtype=np.uint8)
    # High mutation probability ensures some flips
    bit_flip_mutation(population, 0.9, rng)
    # Should have some 1s now
    assert np.sum(population) > 0


def test_bit_flip_mutation_zero_probability(binary_population):
    """Zero mutation probability makes no changes."""
    original = binary_population.copy()
    rng = np.random.default_rng(42)
    bit_flip_mutation(binary_population, 0.0, rng)
    np.testing.assert_array_equal(binary_population, original)


def test_bit_flip_mutation_preserves_binary(binary_population, rng):
    """Mutation keeps values in {0, 1}."""
    bit_flip_mutation(binary_population, 0.5, rng)
    assert np.all((binary_population == 0) | (binary_population == 1))


def test_bit_flip_mutation_respects_probability(rng):
    """Mutation probability roughly matches expected flip rate."""
    population = np.zeros((100, 100), dtype=np.uint8)
    mutation_prob = 0.1
    bit_flip_mutation(population, mutation_prob, rng)
    flip_rate = np.sum(population) / population.size
    # Should be roughly 10% (allow some variance)
    assert 0.05 < flip_rate < 0.15
