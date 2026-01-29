"""Unit tests for parent pairing strategies."""

import numpy as np
import pytest
from src.ga_core.strategies.parent_pairing import (
    random_pairing,
    sequential_pairing,
)


@pytest.fixture
def parent_pool():
    """Sample parent pool with 10 parents."""
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


def test_random_pairing_shape(parent_pool, rng):
    """Random pairing creates correct number of pairs."""
    result = random_pairing(rng, parent_pool)
    assert result.shape == (5, 2)


def test_random_pairing_uses_all_parents(parent_pool, rng):
    """Random pairing uses all parents exactly once."""
    result = random_pairing(rng, parent_pool)
    all_indices = set(result.flatten())
    assert all_indices == set(parent_pool)


def test_sequential_pairing_shape(parent_pool, rng):
    """Sequential pairing creates correct number of pairs."""
    result = sequential_pairing(rng, parent_pool)
    assert result.shape == (5, 2)


def test_sequential_pairing_preserves_order(parent_pool, rng):
    """Sequential pairing maintains input order."""
    result = sequential_pairing(rng, parent_pool)
    expected = np.array(parent_pool).reshape(-1, 2)
    np.testing.assert_array_equal(result, expected)


def test_sequential_pairing_deterministic(parent_pool):
    """Sequential pairing is deterministic."""
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(999)
    result1 = sequential_pairing(rng1, parent_pool)
    result2 = sequential_pairing(rng2, parent_pool)
    np.testing.assert_array_equal(result1, result2)
