"""Binary encoding strategies for genetic algorithms.

Provides binary-specific implementations of crossover, mutation,
and fitness evaluation for the 0-1 knapsack problem.
"""

from src.ga_core.strategies.binary.binary_crossover_kernel import (
    double_point_crossover,
    single_point_crossover,
)
from src.ga_core.strategies.binary.fitness_score import fitness_calculation
from src.ga_core.strategies.binary.mutation import bit_flip_mutation
from src.ga_core.strategies.binary.reproduction_builder import (
    create_binary_reproduction_executor,
)

__all__ = [
    "single_point_crossover",
    "double_point_crossover",
    "bit_flip_mutation",
    "fitness_calculation",
    "create_binary_reproduction_executor",
]
