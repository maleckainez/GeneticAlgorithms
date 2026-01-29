"""Crossover and mutation helpers for binary encoding."""

import numpy as np

from src.ga_core.config.experiment_config import CrossoverType
from src.ga_core.engine.population.types import PopulationType
from src.ga_core.strategies.binary.binary_crossover_kernel import (
    double_point_crossover,
    single_point_crossover,
)
from src.ga_core.strategies.binary.mutation import bit_flip_mutation
from src.ga_core.strategies.protocols import ReproductionExecutor


def create_binary_reproduction_executor(
    crossover_type: CrossoverType,
    crossover_probability: float,
    mutation_probability: float,
) -> ReproductionExecutor:
    """Create reproduction executor for binary encoding.

    Factory function that configures crossover and mutation strategies
    for binary-encoded genetic algorithms.

    Args:
        crossover_type: Type of crossover ("one" or "two" point).
        crossover_probability: Probability of crossover per parent pair.
        mutation_probability: Probability of mutation per gene.

    Returns:
        Callable that executes reproduction (crossover + mutation).
    """
    crossover_kernel = (
        single_point_crossover
        if crossover_type == CrossoverType.ONE_POINT
        else double_point_crossover
    )

    def reproduction_executor(
        population: PopulationType,
        children: PopulationType,
        rng: np.random.Generator,
        parent_pairs: np.ndarray,
    ) -> None:
        """Execute crossover and mutation on parent pairs.

        Operates in-place on children array. Processes all parent pairs
        at once (vectorized operations).

        Args:
            population: Parent population array.
            children: Children array to write results.
            rng: Random number generator.
            parent_pairs: Array of shape (n_pairs, 2) with parent indices.
        """
        p1 = population[parent_pairs[:, 0]]
        p2 = population[parent_pairs[:, 1]]
        c1, c2 = p1.copy(), p2.copy()

        # Apply crossover
        mask = rng.random(size=len(parent_pairs)) < crossover_probability
        c1, c2 = crossover_kernel(c1, c2, p1, p2, mask, rng)

        # Concatenate children and apply mutation
        children[:] = np.concatenate((c1, c2), axis=0)
        if mutation_probability > 0:
            bit_flip_mutation(children, mutation_probability, rng)

    return reproduction_executor
