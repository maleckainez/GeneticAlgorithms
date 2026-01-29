"""Unit tests for strategy factory functions."""

import numpy as np
import pytest
from src.ga_core.config.experiment_config import (
    CrossoverType,
    EncodingType,
    SelectionType,
)
from src.ga_core.strategies.factory import (
    create_elitism_fn,
    create_fitness_fn,
    create_pairing_fn,
    create_reproduction_executor,
    create_selection_fn,
)


def test_create_selection_fn_roulette():
    """Factory creates roulette selection function."""
    fn = create_selection_fn(SelectionType.ROULETTE, population_size=10)
    assert callable(fn)
    fitness = np.array([[100, 50], [80, 40], [60, 30]], dtype=np.int64)
    rng = np.random.default_rng(42)
    result = fn(fitness, 10, rng)
    assert len(result) == 10


def test_create_selection_fn_tournament():
    """Factory creates tournament selection function."""
    fn = create_selection_fn(
        SelectionType.TOURNAMENT, population_size=10, tournament_size=3
    )
    assert callable(fn)
    fitness = np.array([[100, 50], [80, 40], [60, 30]], dtype=np.int64)
    rng = np.random.default_rng(42)
    result = fn(fitness, 10, rng)
    assert len(result) == 10


def test_create_selection_fn_rank():
    """Factory creates rank selection function."""
    fn = create_selection_fn(
        SelectionType.LINEAR_RANK, population_size=10, selection_pressure=1.5
    )
    assert callable(fn)
    fitness = np.array([[100, 50], [80, 40], [60, 30]], dtype=np.int64)
    rng = np.random.default_rng(42)
    result = fn(fitness, 10, rng)
    assert len(result) == 10


def test_create_pairing_fn_random():
    """Factory creates random pairing function."""
    fn = create_pairing_fn("random")
    assert callable(fn)
    parent_pool = [0, 1, 2, 3, 4, 5]
    rng = np.random.default_rng(42)
    result = fn(rng, parent_pool)
    assert result.shape == (3, 2)


def test_create_pairing_fn_sequential():
    """Factory creates sequential pairing function."""
    fn = create_pairing_fn("sequential")
    assert callable(fn)
    parent_pool = [0, 1, 2, 3, 4, 5]
    rng = np.random.default_rng(42)
    result = fn(rng, parent_pool)
    assert result.shape == (3, 2)


def test_create_pairing_fn_invalid():
    """Factory raises error for invalid pairing strategy."""
    with pytest.raises(ValueError, match="Unknown pairing strategy"):
        create_pairing_fn("invalid")


def test_create_elitism_fn():
    """Factory creates elitism function."""
    fn = create_elitism_fn()
    assert callable(fn)
    fitness = np.array([[100, 50], [80, 40], [60, 30]], dtype=np.int64)
    result = fn(fitness, 0.5)
    assert len(result) == 1  # 50% of 3 = 1


def test_create_reproduction_executor_binary():
    """Factory creates binary reproduction executor."""
    executor = create_reproduction_executor(
        EncodingType.BINARY, CrossoverType.ONE_POINT, 0.8, 0.01
    )
    assert callable(executor)


def test_create_reproduction_executor_unsupported():
    """Factory raises error for unsupported encoding."""
    with pytest.raises(ValueError, match="Unsupported encoding type"):
        # Use a string that's not a valid enum value
        create_reproduction_executor("invalid", CrossoverType.ONE_POINT, 0.8, 0.01)


def test_create_fitness_fn_binary():
    """Factory creates binary fitness function."""
    fn = create_fitness_fn(EncodingType.BINARY)
    assert callable(fn)


def test_create_fitness_fn_unsupported():
    """Factory raises error for unsupported encoding."""
    with pytest.raises(ValueError, match="Unsupported encoding type"):
        create_fitness_fn("invalid")
