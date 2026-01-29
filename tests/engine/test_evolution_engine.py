"""Simple unit tests for EvolutionEngine."""

import numpy as np
from src.ga_core.engine.strategies.executor import StrategyExecutor

# Simple hardcoded test data
TEST_CONFIG_DICT = {
    "data": {"data_filename": "test.txt", "max_weight": 10},
    "population": {"size": 10, "generations": 10},
    "selection": {"type": "rank", "selection_pressure": 1.5, "tournament_size": None},
    "genetic_operators": {
        "crossover_type": "one",
        "crossover_probability": 0.6,
        "mutation_probability": 0.1,
        "penalty_multiplier": 2.0,
    },
    "experiment": {"seed": 42, "identifier": "test", "log_level": "INFO"},
}


def test_strategy_executor_initializes():
    """StrategyExecutor can be created with simple lambdas."""
    executor = StrategyExecutor(
        selection_fn=lambda pop, fit, n, rng: np.arange(min(n, len(pop))),
        pairing_fn=lambda parents, rng: [(0, 1)] if len(parents) >= 2 else [],
        reproduction_executor=lambda pop, pairs, rng: pop[:2],
        fitness_fn=lambda pop: np.ones(len(pop)),
        population_size=10,
    )

    assert executor.population_size == 10


def test_strategy_executor_evaluates_fitness():
    """StrategyExecutor evaluates fitness on population."""

    def simple_fitness(pop):
        return np.sum(pop, axis=1).astype(float)

    executor = StrategyExecutor(
        selection_fn=lambda pop, fit, n, rng: np.arange(min(n, len(pop))),
        pairing_fn=lambda parents, rng: [(0, 1)],
        reproduction_executor=lambda pop, pairs, rng: pop[:2],
        fitness_fn=simple_fitness,
        population_size=4,
    )

    population = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    fitness = executor.evaluate_fitness(population)

    assert len(fitness) == 4
    assert fitness[0] == 3.0
    assert fitness[1] == 0.0


def test_strategy_executor_executes_generation():
    """StrategyExecutor runs full generation pipeline."""
    executor = StrategyExecutor(
        selection_fn=lambda fit, n, rng: list(range(min(n, len(fit)))),
        pairing_fn=lambda rng, parents: (
            [(parents[0], parents[1])] if len(parents) >= 2 else []
        ),
        reproduction_executor=lambda pop, children, rng, pairs: None,
        fitness_fn=lambda pop: np.ones(len(pop)),
        population_size=4,
    )

    population = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    children = np.zeros((4, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)

    selected_idx, parent_pop = executor.execute_generation(population, children, rng)

    assert len(selected_idx) > 0
