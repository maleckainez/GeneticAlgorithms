from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    data_filename: str
    population_size: int
    generations: int
    max_weight: int
    seed: int | None
    crossover_probability: float
    mutation_probability: float
    penalty: float
    experiment_identifier: int
    stream_batch_size: int | None = None
    rng: np.random.Generator | None = None

    def __post_init__(self):
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

    def generate_probability_of_failure(self, weight_sum: int) -> float:
        if weight_sum < 1:
            raise ValueError("Weight sum must be greater than 0")
        probability_of_failure = self.max_weight / weight_sum
        return max(0, min(1.0, probability_of_failure))
