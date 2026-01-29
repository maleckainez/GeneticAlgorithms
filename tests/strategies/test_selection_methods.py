"""Unit tests for selection methods."""

import numpy as np
import pytest
from src.ga_core.strategies.selection_methods import (
    linear_rank_selection,
    roulette_selection,
    tournament_selection,
)


@pytest.fixture
def fitness_array():
    """Sample fitness array (fitness, weight)."""
    return np.array([[100, 50], [80, 40], [60, 30], [40, 20], [20, 10]], dtype=np.int64)


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


def test_roulette_selection_returns_correct_count(fitness_array, rng):
    """Roulette selection returns requested number of parents."""
    result = roulette_selection(fitness_array, 10, rng)
    assert len(result) == 10
    assert all(0 <= idx < len(fitness_array) for idx in result)


def test_roulette_selection_zero_fitness(rng):
    """Roulette selection handles zero fitness with pseudo-fitness."""
    zero_fitness = np.array([[0, 50], [0, 40], [0, 30]], dtype=np.int64)
    result = roulette_selection(zero_fitness, 6, rng)
    assert len(result) == 6


def test_tournament_selection_returns_correct_count(fitness_array, rng):
    """Tournament selection returns requested number of parents."""
    result = tournament_selection(fitness_array, 10, rng, tournament_size=3)
    assert len(result) == 10
    assert all(0 <= idx < len(fitness_array) for idx in result)


def test_tournament_selection_selects_best(fitness_array):
    """Tournament selection prefers higher fitness individuals."""
    rng = np.random.default_rng(42)
    result = tournament_selection(fitness_array, 20, rng, tournament_size=5)
    # Best individual (index 0) should appear frequently
    assert result.count(0) > 0


def test_linear_rank_selection_returns_correct_count(fitness_array, rng):
    """Rank selection returns requested number of parents."""
    result = linear_rank_selection(fitness_array, 10, rng, selection_pressure=1.5)
    assert len(result) == 10
    assert all(0 <= idx < len(fitness_array) for idx in result)


def test_linear_rank_selection_pressure_bounds(fitness_array, rng):
    """Rank selection works with different pressure values."""
    result_low = linear_rank_selection(fitness_array, 10, rng, selection_pressure=1.0)
    result_high = linear_rank_selection(fitness_array, 10, rng, selection_pressure=2.0)
    assert len(result_low) == 10
    assert len(result_high) == 10
