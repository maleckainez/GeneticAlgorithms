"""Unit tests for elitism strategy."""

import numpy as np
import pytest
from src.ga_core.strategies.elitism import select_elites


@pytest.fixture
def fitness_array():
    """Sample fitness array (fitness, weight)."""
    return np.array([[100, 50], [80, 40], [60, 30], [40, 20], [20, 10]], dtype=np.int64)


def test_select_elites_returns_correct_count(fitness_array):
    """Elite selection returns correct number of individuals."""
    result = select_elites(fitness_array, elite_percentage=0.4)
    assert len(result) == 2  # 40% of 5 = 2


def test_select_elites_selects_best(fitness_array):
    """Elite selection picks individuals with highest fitness."""
    result = select_elites(fitness_array, elite_percentage=0.4)
    # Should select indices 0 and 1 (best two)
    assert 0 in result
    assert 1 in result


def test_select_elites_zero_percentage(fitness_array):
    """Elite selection with 0% returns empty array."""
    result = select_elites(fitness_array, elite_percentage=0.0)
    assert len(result) == 0


def test_select_elites_full_percentage(fitness_array):
    """Elite selection with 100% returns all individuals."""
    result = select_elites(fitness_array, elite_percentage=1.0)
    assert len(result) == len(fitness_array)


def test_select_elites_sorted_descending(fitness_array):
    """Elite indices are sorted by fitness (best first)."""
    result = select_elites(fitness_array, elite_percentage=0.6)
    # Extract fitness values of selected elites
    elite_fitness = fitness_array[result, 0]
    # Should be in descending order
    assert all(
        elite_fitness[i] >= elite_fitness[i + 1] for i in range(len(elite_fitness) - 1)
    )
