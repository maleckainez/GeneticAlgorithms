"""Unit tests for engine strategy executor."""

import numpy as np
import pytest
from src.ga_core.engine.strategies.executor import StrategyExecutor


@pytest.fixture
def mock_strategies():
    """Create mock strategies for testing."""

    def selection_fn(fitness_arr, population_size, rng):
        """Mock selection - returns indices 0,1,0,1..."""
        return [i % 2 for i in range(population_size)]

    def pairing_fn(rng, parent_pool):
        """Mock pairing - sequential pairs."""
        return np.array(parent_pool).reshape(-1, 2)

    def reproduction_executor(population, children, rng, parent_pairs):
        """Mock reproduction - copy parents to children."""
        for i, (p1_idx, p2_idx) in enumerate(parent_pairs):
            children[i * 2] = population[p1_idx]
            children[i * 2 + 1] = population[p2_idx]

    def fitness_fn(population, **kwargs):
        """Mock fitness - sum of genes."""
        fitness = population.sum(axis=1)
        weight = population.sum(axis=1)
        return np.column_stack([fitness, weight])

    return selection_fn, pairing_fn, reproduction_executor, fitness_fn


@pytest.fixture
def sample_population():
    """Sample binary population."""
    return np.array(
        [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]],
        dtype=np.uint8,
    )


@pytest.fixture
def rng():
    """Seeded RNG."""
    return np.random.default_rng(42)


def test_strategy_executor_initialization(mock_strategies):
    """StrategyExecutor initializes with strategies."""
    selection_fn, pairing_fn, reproduction_executor, fitness_fn = mock_strategies

    executor = StrategyExecutor(
        selection_fn=selection_fn,
        pairing_fn=pairing_fn,
        reproduction_executor=reproduction_executor,
        fitness_fn=fitness_fn,
        population_size=4,
    )

    assert executor.selection_fn == selection_fn
    assert executor.pairing_fn == pairing_fn
    assert executor.reproduction_executor == reproduction_executor
    assert executor.fitness_fn == fitness_fn
    assert executor.population_size == 4


def test_strategy_executor_execute_generation(mock_strategies, sample_population, rng):
    """Execute generation runs complete pipeline."""
    selection_fn, pairing_fn, reproduction_executor, fitness_fn = mock_strategies

    executor = StrategyExecutor(
        selection_fn=selection_fn,
        pairing_fn=pairing_fn,
        reproduction_executor=reproduction_executor,
        fitness_fn=fitness_fn,
        population_size=4,
    )

    children = np.zeros_like(sample_population)
    parent_indices, fitness_array = executor.execute_generation(
        population=sample_population, children=children, rng=rng
    )

    # Check parent indices
    assert len(parent_indices) == 4
    assert parent_indices == [0, 1, 0, 1]

    # Check fitness array
    assert fitness_array.shape == (4, 2)

    # Check children were populated
    assert not np.all(children == 0)


def test_strategy_executor_evaluate_fitness(mock_strategies, sample_population):
    """Evaluate fitness returns fitness array."""
    selection_fn, pairing_fn, reproduction_executor, fitness_fn = mock_strategies

    executor = StrategyExecutor(
        selection_fn=selection_fn,
        pairing_fn=pairing_fn,
        reproduction_executor=reproduction_executor,
        fitness_fn=fitness_fn,
        population_size=4,
    )

    fitness_array = executor.evaluate_fitness(sample_population)

    assert fitness_array.shape == (4, 2)
    # First individual (all 1s) should have fitness 5
    assert fitness_array[0, 0] == 5
    # Second individual (all 0s) should have fitness 0
    assert fitness_array[1, 0] == 0


def test_strategy_executor_with_fitness_kwargs(mock_strategies, sample_population, rng):
    """Execute generation passes fitness kwargs."""

    def fitness_with_kwargs(population, max_weight=100, **kwargs):
        """Fitness that uses kwargs."""
        fitness = population.sum(axis=1)
        weight = population.sum(axis=1)
        # Apply some transformation based on max_weight
        adjusted_fitness = np.where(weight > max_weight / 10, 0, fitness)
        return np.column_stack([adjusted_fitness, weight])

    selection_fn, pairing_fn, reproduction_executor, _ = mock_strategies

    executor = StrategyExecutor(
        selection_fn=selection_fn,
        pairing_fn=pairing_fn,
        reproduction_executor=reproduction_executor,
        fitness_fn=fitness_with_kwargs,
        population_size=4,
    )

    children = np.zeros_like(sample_population)
    parent_indices, fitness_array = executor.execute_generation(
        population=sample_population, children=children, rng=rng, max_weight=100
    )

    assert fitness_array.shape == (4, 2)
