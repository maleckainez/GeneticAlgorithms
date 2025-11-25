"""Module for managing genetic algorithm configuration parameters."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration dataclass for a single run of the genetic algorithm.

    It holds all necessary parameters (probabilities, sizes, types, and resource
    handles) and performs critical validation checks upon instantiation.

    Attributes:
        data_filename (str): Name of the input file.
        population_size (int): Number of individuals in the population (must be even).
        generations (int): Maximum number of generations to run.
        max_weight (int): The maximum allowed weight.
        seed (int | None): Seed for the random number generator.
        selection_type (str): Type of selection method (`roulette`,
                              `tournament`, `rank`).
        crossover_type (str): Type of crossover (`one`, `two`).
        crossover_probability (float): Probability of performing crossover (0 to 1).
        mutation_probability (float): Probability of mutation per gene (0 to 1).
        penalty (float): Penalty factor for exceeding the maximum weight.
        experiment_identifier (int): Unique ID for the experiment run.
        log_level (str): Logging verbosity level.
        stream_batch_size (int | None): Size of the stream batches for memmap I/O.
                                        Defaults to `500` if `None`.
        rng (np.random.Generator | None): NumPy random generator instance.
                                          Initialized if `None`.
        selection_pressure (float | None): Selection pressure (1.0 to 2.0) used
                                           in `rank` selection. Defaults to `1.0`.
    """

    data_filename: str
    population_size: int
    generations: int
    max_weight: int
    seed: int | None
    selection_type: str
    crossover_type: str
    crossover_probability: float
    mutation_probability: float
    penalty: float
    experiment_identifier: int
    log_level: str
    stream_batch_size: int | None = None
    rng: np.random.Generator | None = None
    selection_pressure: float | None = None

    def __post_init__(self) -> None:
        """Performs validation checks.

        Raises:
            ValueError: If any parameter fails the validation checks.
        """
        if self.population_size % 2 != 0:
            raise ValueError("Population has to be even!")
        if self.crossover_probability > 1 or self.crossover_probability < 0:
            raise ValueError("Crossover_probability must be between 0 and 1")
        if self.mutation_probability > 1 or self.mutation_probability < 0:
            raise ValueError("Mutation probability must be between 0 and 1")
        if self.max_weight < 0:
            raise ValueError("Max weight must be positive and non-zero")
        if self.population_size < 1:
            raise ValueError("Population size must be greater than 0")
        if self.generations < 1:
            raise ValueError("Generations must be greater than 1")
        if self.stream_batch_size is None or self.stream_batch_size < 1:
            object.__setattr__(self, "stream_batch_size", 500)
        if self.rng is None:
            if self.seed is None:
                object.__setattr__(self, "rng", np.random.default_rng())
            else:
                object.__setattr__(self, "rng", np.random.default_rng(self.seed))
        if self.selection_type == "rank":
            if self.selection_pressure is None:
                object.__setattr__(self, "selection_pressure", 1.0)
            assert self.selection_pressure is not None
            if self.selection_pressure < 1 or self.selection_pressure > 2:
                raise ValueError(
                    "Selection pressure must be float in range from 1 to 2"
                )

    def generate_probability_of_failure(self, weight_sum: int) -> float:
        """Calculates the maximum probability of failure (1 - P(Success)).

        This value is used to bias initial population generation.

        Args:
            weight_sum (int): The total possible weight of all items combined.

        Raises:
            ValueError: If the weight sum is less than 1.

        Returns:
            float: The probability of failure (clamped between 0.0 and 1.0).
        """
        if weight_sum < 1:
            raise ValueError("Weight sum must be greater than 0")
        probability_of_failure = self.max_weight / weight_sum
        return max(0, min(1.0, probability_of_failure))
