"""Strategy execution types and protocols for evolution engine."""

from typing import Protocol

import numpy as np

from src.ga_core.engine.population.types import PopulationType


class StrategyExecutor(Protocol):
    """Protocol for executing a complete evolution generation step."""

    def execute_generation(
        self,
        population: PopulationType,
        children: PopulationType,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute one generation: selection → pairing → reproduction.

        Args:
            population: Current population array.
            children: Buffer for offspring.
            rng: Random number generator.

        Returns:
            Tuple of (parent_indices, fitness_array).
        """
        ...
