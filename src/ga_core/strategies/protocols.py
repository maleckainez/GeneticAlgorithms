"""Protocol definitions for genetic algorithm strategies.

This module defines the contracts for all strategy functions used in the GA,
enabling type checking and ensuring consistent interfaces across implementations.
"""

from typing import Protocol, Sequence

import numpy as np

from src.ga_core.engine.population.types import PopulationType


class ReproductionExecutor(Protocol):
    """Protocol for reproduction executor (crossover + mutation).

    Combines crossover and mutation operations into a single callable
    that modifies the children array in-place.
    """

    def __call__(
        self,
        population: PopulationType,
        children: PopulationType,
        rng: np.random.Generator,
        parent_pairs: np.ndarray,
    ) -> None:
        """Execute reproduction on parent pairs.

        Args:
            population: Parent population array.
            children: Buffer for offspring (modified in-place).
            rng: Random number generator.
            parent_pairs: Array of shape (n_pairs, 2) with parent indices.
        """
        ...


class SelectionFn(Protocol):
    """Protocol for parent selection function.

    Selects individuals from population based on fitness for reproduction.
    """

    def __call__(
        self,
        fitness_arr: np.ndarray,
        population_size: int,
        rng: np.random.Generator,
        *args,
        **kwargs,
    ) -> Sequence[int]:
        """Select parents based on fitness.

        Args:
            fitness_arr: Array of shape (pop_size, 2) with fitness and weight.
            population_size: Number of parents to select.
            rng: Random number generator.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Sequence of selected parent indices.
        """
        ...


class FitnessFn(Protocol):
    """Protocol for fitness evaluation function.

    Evaluates fitness of all individuals in the population.
    """

    def __call__(self, population: PopulationType, *args, **kwargs) -> np.ndarray:
        """Evaluate fitness of population.

        Args:
            population: Population array to evaluate.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Array of shape (pop_size, 2) with fitness and weight values.
        """
        ...


class PairingFn(Protocol):
    """Protocol for parent pairing function.

    Pairs selected parents for reproduction.
    """

    def __call__(
        self, rng: np.random.Generator, parent_pool: Sequence[int]
    ) -> np.ndarray:
        """Pair parents from parent pool.

        Args:
            rng: Random number generator.
            parent_pool: Sequence of parent indices to pair.

        Returns:
            Array of shape (n_pairs, 2) with paired parent indices.
        """
        ...


class ElitismFn(Protocol):
    """Protocol for elitism selection function.

    Selects elite individuals to preserve across generations.
    """

    def __call__(self, fitness_arr: np.ndarray, elite_percentage: float) -> np.ndarray:
        """Select elite individuals.

        Args:
            fitness_arr: Array of shape (pop_size, 2) with fitness and weight.
            elite_percentage: Fraction of population to preserve (0.0-1.0).

        Returns:
            Array of elite individual indices, sorted by fitness.
        """
        ...
