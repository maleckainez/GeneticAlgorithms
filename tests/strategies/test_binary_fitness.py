"""Unit tests for binary fitness calculation."""

import numpy as np
import pytest
from src.ga_core.strategies.binary.fitness_score import fitness_calculation


@pytest.fixture
def knapsack_data():
    """Sample knapsack items (value, weight)."""
    value_arr = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    weight_arr = np.array([5, 10, 15, 20, 25], dtype=np.int64)
    return value_arr, weight_arr


@pytest.fixture
def binary_population():
    """Sample binary population."""
    return np.array([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 1]], dtype=np.uint8)


def test_fitness_calculation_shape(binary_population, knapsack_data):
    """Fitness calculation returns correct shape."""
    value_arr, weight_arr = knapsack_data
    result = fitness_calculation(
        max_weight=50,
        penalty_factor=1.0,
        population=binary_population,
        batch_size=10,
        value_arr=value_arr,
        weight_arr=weight_arr,
    )
    assert result.shape == (3, 2)  # (population_size, 2)


def test_fitness_calculation_computes_values(binary_population, knapsack_data):
    """Fitness calculation computes correct values and weights."""
    value_arr, weight_arr = knapsack_data
    result = fitness_calculation(
        max_weight=100,
        penalty_factor=1.0,
        population=binary_population,
        batch_size=10,
        value_arr=value_arr,
        weight_arr=weight_arr,
    )
    # First individual: items 0,1 -> value=30, weight=15
    assert result[0, 0] == 30
    assert result[0, 1] == 15
    # Second individual: items 0,1,2 -> value=60, weight=30
    assert result[1, 0] == 60
    assert result[1, 1] == 30
    # Third individual: item 4 -> value=50, weight=25
    assert result[2, 0] == 50
    assert result[2, 1] == 25


def test_fitness_calculation_applies_penalty(knapsack_data):
    """Fitness calculation penalizes overweight solutions."""
    value_arr, weight_arr = knapsack_data
    # Create overweight individual: all items
    population = np.array([[1, 1, 1, 1, 1]], dtype=np.uint8)
    # Total weight = 75, max_weight = 50
    result = fitness_calculation(
        max_weight=50,
        penalty_factor=2.0,
        population=population,
        batch_size=10,
        value_arr=value_arr,
        weight_arr=weight_arr,
    )
    # Total value = 150, overweight = 25, penalty = 25 * 2 = 50
    # Penalized fitness = 150 - 50 = 100
    assert result[0, 0] == 100
    assert result[0, 1] == 75


def test_fitness_calculation_zero_penalty_nullifies(knapsack_data):
    """Zero penalty factor nullifies overweight fitness to zero."""
    value_arr, weight_arr = knapsack_data
    population = np.array([[1, 1, 1, 1, 1]], dtype=np.uint8)
    result = fitness_calculation(
        max_weight=50,
        penalty_factor=0.0,  # Zero penalty
        population=population,
        batch_size=10,
        value_arr=value_arr,
        weight_arr=weight_arr,
    )
    # With zero penalty_factor, overweight fitness is set to zero
    assert result[0, 0] == 0
    assert result[0, 1] == 75


def test_fitness_calculation_batched_processing(binary_population, knapsack_data):
    """Fitness calculation works with small batch sizes."""
    value_arr, weight_arr = knapsack_data
    # Use batch size of 1 to test streaming
    result = fitness_calculation(
        max_weight=100,
        penalty_factor=1.0,
        population=binary_population,
        batch_size=1,
        value_arr=value_arr,
        weight_arr=weight_arr,
    )
    assert result.shape == (3, 2)
    # Results should be same as non-batched
    assert result[0, 0] == 30
    assert result[1, 0] == 60
    assert result[2, 0] == 50
