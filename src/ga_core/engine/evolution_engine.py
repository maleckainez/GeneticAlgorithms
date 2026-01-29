"""Core evolution engine for running genetic algorithm generations.

This module provides the main execution loop for genetic algorithms,
orchestrating strategy execution, population management, and state tracking.
"""

from typing import Optional

import numpy as np

from src.ga_core.config.experiment_config import ExperimentConfig
from src.ga_core.engine.population.types import PopulationType
from src.ga_core.engine.strategies import StrategyExecutor
from src.ga_core.engine.timer import Timer


class GenerationStats:
    """Statistics from one generation of evolution."""

    def __init__(
        self,
        iteration: int,
        fitness_array: np.ndarray,
        parent_indices: list[int],
    ):
        """Initialize generation statistics.

        Args:
            iteration: Generation number.
            fitness_array: Fitness values for all individuals.
            parent_indices: Indices of selected parents.
        """
        self.iteration = iteration
        self.fitness_array = fitness_array
        self.parent_indices = parent_indices

        # Calculate statistics (sort fitness descending, weight ascending)
        sort_keys = np.lexsort((fitness_array[:, 1], fitness_array[:, 0]))
        self.best_idx = int(sort_keys[-1])  # Last is highest fitness
        self.best_fitness = int(fitness_array[self.best_idx, 0])
        self.best_weight = int(fitness_array[self.best_idx, 1])
        self.avg_fitness = float(np.mean(fitness_array[:, 0]))
        self.worst_idx = int(sort_keys[0])  # First is lowest fitness
        self.worst_fitness = int(fitness_array[self.worst_idx, 0])
        self.worst_weight = int(fitness_array[self.worst_idx, 1])

        # Count identical best solutions
        mask = (fitness_array[:, 0] == self.best_fitness) & (
            fitness_array[:, 1] == self.best_weight
        )
        self.identical_best_count = int(np.sum(mask)) - 1


class EvolutionEngine:
    """Main evolution engine for genetic algorithms.

    Orchestrates the complete evolution process including strategy execution,
    population swapping, and statistics tracking. Designed to work with
    pluggable strategies and population managers.
    """

    def __init__(
        self,
        strategy_executor: StrategyExecutor,
        config: ExperimentConfig,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize evolution engine.

        Args:
            strategy_executor: Configured strategy executor.
            config: Experiment configuration.
            rng: Random number generator (default: create from config seed).
        """
        self.strategy_executor = strategy_executor
        self.config = config
        self.rng = rng if rng is not None else config.create_rng()
        self.timer = Timer(config_like=config, logger=None)
        self.current_generation = 0

    def run_generation(
        self,
        population: PopulationType,
        children: PopulationType,
        **fitness_kwargs,
    ) -> GenerationStats:
        """Execute one complete generation of evolution.

        Args:
            population: Current population array.
            children: Buffer for offspring (will be modified).
            **fitness_kwargs: Additional arguments for fitness evaluation.

        Returns:
            Statistics from this generation.
        """
        self.current_generation += 1

        # Execute strategy pipeline
        parent_indices, fitness_array = self.strategy_executor.execute_generation(
            population=population,
            children=children,
            rng=self.rng,
            **fitness_kwargs,
        )

        # Create stats
        stats = GenerationStats(
            iteration=self.current_generation,
            fitness_array=fitness_array,
            parent_indices=parent_indices,
        )

        return stats

    def evaluate_initial_population(
        self, population: PopulationType, **fitness_kwargs
    ) -> GenerationStats:
        """Evaluate fitness of initial population (generation 0).

        Args:
            population: Initial population array.
            **fitness_kwargs: Additional arguments for fitness evaluation.

        Returns:
            Statistics for generation 0.
        """
        fitness_array = self.strategy_executor.evaluate_fitness(
            population, **fitness_kwargs
        )

        stats = GenerationStats(
            iteration=0, fitness_array=fitness_array, parent_indices=[]
        )

        return stats

    def run_evolution(
        self,
        population: PopulationType,
        children: PopulationType,
        generations: int,
        fitness_kwargs: Optional[dict] = None,
    ) -> list[GenerationStats]:
        """Run complete evolution for specified number of generations.

        Args:
            population: Initial population array.
            children: Buffer for offspring.
            generations: Number of generations to run.
            fitness_kwargs: Additional arguments for fitness evaluation.

        Returns:
            List of statistics for each generation.
        """
        if fitness_kwargs is None:
            fitness_kwargs = {}

        all_stats = []

        # Evaluate initial population
        initial_stats = self.evaluate_initial_population(population, **fitness_kwargs)
        all_stats.append(initial_stats)

        # Run generations
        for gen in range(generations):
            stats = self.run_generation(population, children, **fitness_kwargs)
            all_stats.append(stats)

            # Swap populations (children become new population)
            # NOTE: In real implementation, this would be handled by PopulationManager
            # For now, assume caller handles the swap
            pass

        return all_stats
