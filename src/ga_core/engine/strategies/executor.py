"""Strategy execution orchestrator for genetic algorithm engine.

Combines selection, pairing, and reproduction strategies into a cohesive
execution pipeline for one generation of evolution.
"""

import numpy as np

from src.ga_core.engine.population.types import PopulationType
from src.ga_core.strategies.protocols import (
    FitnessFn,
    PairingFn,
    ReproductionExecutor,
    SelectionFn,
)


class StrategyExecutor:
    """Orchestrates strategy execution for one generation.

    Combines selection, pairing, and reproduction into a single
    execution pipeline. Handles fitness evaluation and returns
    statistics for logging.
    """

    def __init__(
        self,
        selection_fn: SelectionFn,
        pairing_fn: PairingFn,
        reproduction_executor: ReproductionExecutor,
        fitness_fn: FitnessFn,
        population_size: int,
    ):
        """Initialize strategy executor with configured strategies.

        Args:
            selection_fn: Parent selection strategy.
            pairing_fn: Parent pairing strategy.
            reproduction_executor: Crossover and mutation executor.
            fitness_fn: Fitness evaluation function.
            population_size: Size of population.
        """
        self.selection_fn = selection_fn
        self.pairing_fn = pairing_fn
        self.reproduction_executor = reproduction_executor
        self.fitness_fn = fitness_fn
        self.population_size = population_size

    def execute_generation(
        self,
        population: PopulationType,
        children: PopulationType,
        rng: np.random.Generator,
        **fitness_kwargs,
    ) -> tuple[list[int], np.ndarray]:
        """Execute one complete generation cycle.

        Pipeline:
        1. Evaluate fitness of current population
        2. Select parents based on fitness
        3. Pair parents for reproduction
        4. Execute reproduction (crossover + mutation)

        Args:
            population: Current population array.
            children: Buffer for offspring (modified in-place).
            rng: Random number generator.
            **fitness_kwargs: Additional arguments for fitness function.

        Returns:
            Tuple of (parent_indices, fitness_array).
        """
        # 1. Evaluate fitness
        fitness_array = self.fitness_fn(population, **fitness_kwargs)

        # 2. Select parents
        parent_indices = self.selection_fn(fitness_array, self.population_size, rng)

        # 3. Pair parents
        parent_pairs = self.pairing_fn(rng, parent_indices)

        # 4. Execute reproduction
        self.reproduction_executor(population, children, rng, parent_pairs)

        return parent_indices, fitness_array

    def evaluate_fitness(
        self, population: PopulationType, **fitness_kwargs
    ) -> np.ndarray:
        """Evaluate fitness of population.

        Args:
            population: Population array to evaluate.
            **fitness_kwargs: Additional arguments for fitness function.

        Returns:
            Fitness array of shape (population_size, 2) with [fitness, weight].
        """
        return self.fitness_fn(population, **fitness_kwargs)
