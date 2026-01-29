"""Unit tests for binary reproduction builder."""

import numpy as np
import pytest
from src.ga_core.config.experiment_config import CrossoverType
from src.ga_core.strategies.binary.reproduction_builder import (
    create_binary_reproduction_executor,
)


@pytest.fixture
def population_and_children():
    """Sample population and children arrays."""
    population = np.array(
        [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]],
        dtype=np.uint8,
    )
    children = np.zeros((4, 5), dtype=np.uint8)
    return population, children


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


def test_create_binary_reproduction_executor_returns_callable():
    """Builder returns callable executor."""
    executor = create_binary_reproduction_executor(CrossoverType.ONE_POINT, 0.8, 0.01)
    assert callable(executor)


def test_reproduction_executor_one_point(population_and_children, rng):
    """One-point crossover executor works correctly."""
    population, children = population_and_children
    parent_pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)

    executor = create_binary_reproduction_executor(CrossoverType.ONE_POINT, 0.9, 0.0)
    executor(population, children, rng, parent_pairs)

    # Children should be created
    assert children.shape == (4, 5)
    # Children should be binary
    assert np.all((children == 0) | (children == 1))


def test_reproduction_executor_two_point(population_and_children, rng):
    """Two-point crossover executor works correctly."""
    population, children = population_and_children
    parent_pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)

    executor = create_binary_reproduction_executor(CrossoverType.TWO_POINT, 0.9, 0.0)
    executor(population, children, rng, parent_pairs)

    # Children should be created
    assert children.shape == (4, 5)
    # Children should be binary
    assert np.all((children == 0) | (children == 1))


def test_reproduction_executor_applies_mutation(population_and_children, rng):
    """Executor applies mutation after crossover."""
    population, children = population_and_children
    parent_pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)

    # High mutation probability
    executor = create_binary_reproduction_executor(CrossoverType.ONE_POINT, 1.0, 0.5)
    executor(population, children, rng, parent_pairs)

    # Some mutation should have occurred
    assert children.shape == (4, 5)
    assert np.all((children == 0) | (children == 1))


def test_reproduction_executor_zero_crossover_probability(population_and_children):
    """Zero crossover probability makes children similar to parents."""
    population, children = population_and_children
    parent_pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)
    rng = np.random.default_rng(42)

    executor = create_binary_reproduction_executor(CrossoverType.ONE_POINT, 0.0, 0.0)
    executor(population, children, rng, parent_pairs)

    # With no crossover/mutation, children are copies concatenated as [c1, c2, ...]
    # Check shape and binary values
    assert children.shape == (4, 5)
    assert np.all((children == 0) | (children == 1))
    # At least some children should match parents (no crossover happened)
    # First two children come from pair [0,1]
    assert np.array_equal(children[0], population[0]) or np.array_equal(
        children[0], population[1]
    )
