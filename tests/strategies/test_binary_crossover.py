"""Unit tests for binary crossover kernels."""

import numpy as np
import pytest
from src.ga_core.strategies.binary.binary_crossover_kernel import (
    double_point_crossover,
    single_point_crossover,
)


@pytest.fixture
def parent_pairs():
    """Sample parent genomes."""
    p1 = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=np.uint8)
    p2 = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=np.uint8)
    return p1, p2


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


def test_single_point_crossover_shape(parent_pairs, rng):
    """Single-point crossover preserves shape."""
    p1, p2 = parent_pairs
    c1, c2 = p1.copy(), p2.copy()
    mask = np.array([True, True], dtype=bool)
    result_c1, result_c2 = single_point_crossover(c1, c2, p1, p2, mask, rng)
    assert result_c1.shape == p1.shape
    assert result_c2.shape == p2.shape


def test_single_point_crossover_modifies_children(parent_pairs, rng):
    """Single-point crossover creates different children."""
    p1, p2 = parent_pairs
    c1, c2 = p1.copy(), p2.copy()
    mask = np.array([True, True], dtype=bool)
    single_point_crossover(c1, c2, p1, p2, mask, rng)
    # At least one child should differ from its parent
    assert not np.array_equal(c1, p1) or not np.array_equal(c2, p2)


def test_single_point_crossover_respects_mask(parent_pairs, rng):
    """Crossover only affects masked pairs."""
    p1, p2 = parent_pairs
    c1, c2 = p1.copy(), p2.copy()
    mask = np.array([False, False], dtype=bool)  # No crossover
    single_point_crossover(c1, c2, p1, p2, mask, rng)
    # Children should be unchanged
    np.testing.assert_array_equal(c1, p1)
    np.testing.assert_array_equal(c2, p2)


def test_single_point_crossover_preserves_binary(parent_pairs, rng):
    """Crossover keeps values in {0, 1}."""
    p1, p2 = parent_pairs
    c1, c2 = p1.copy(), p2.copy()
    mask = np.array([True, True], dtype=bool)
    single_point_crossover(c1, c2, p1, p2, mask, rng)
    assert np.all((c1 == 0) | (c1 == 1))
    assert np.all((c2 == 0) | (c2 == 1))


def test_double_point_crossover_shape(parent_pairs, rng):
    """Double-point crossover preserves shape."""
    p1, p2 = parent_pairs
    c1, c2 = p1.copy(), p2.copy()
    mask = np.array([True, True], dtype=bool)
    result_c1, result_c2 = double_point_crossover(c1, c2, p1, p2, mask, rng)
    assert result_c1.shape == p1.shape
    assert result_c2.shape == p2.shape


def test_double_point_crossover_modifies_children(parent_pairs, rng):
    """Double-point crossover creates different children."""
    p1, p2 = parent_pairs
    c1, c2 = p1.copy(), p2.copy()
    mask = np.array([True, True], dtype=bool)
    double_point_crossover(c1, c2, p1, p2, mask, rng)
    # At least one child should differ from its parent
    assert not np.array_equal(c1, p1) or not np.array_equal(c2, p2)


def test_double_point_crossover_respects_mask(parent_pairs, rng):
    """Crossover only affects masked pairs."""
    p1, p2 = parent_pairs
    c1, c2 = p1.copy(), p2.copy()
    mask = np.array([False, False], dtype=bool)  # No crossover
    double_point_crossover(c1, c2, p1, p2, mask, rng)
    # Children should be unchanged
    np.testing.assert_array_equal(c1, p1)
    np.testing.assert_array_equal(c2, p2)


def test_double_point_crossover_preserves_binary(parent_pairs, rng):
    """Crossover keeps values in {0, 1}."""
    p1, p2 = parent_pairs
    c1, c2 = p1.copy(), p2.copy()
    mask = np.array([True, True], dtype=bool)
    double_point_crossover(c1, c2, p1, p2, mask, rng)
    assert np.all((c1 == 0) | (c1 == 1))
    assert np.all((c2 == 0) | (c2 == 1))
