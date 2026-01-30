"""Typed experiment configuration and helpers.

This module provides a thin wrapper around :class:`InputConfig` that adds runtime
utilities needed by runners, such as RNG creation and probability estimation for
overweight genomes.
"""

# due to propeties below, ruff must be silenced
# ruff: noqa: D102

from pathlib import Path

from numpy.random import Generator, default_rng
from pydantic import BaseModel

from .input_config_scheme import (
    CrossoverType,
    InputConfig,
    LogLevel,
    SelectionType,
    SwapType,
)


class ExperimentConfig(BaseModel):
    """Runtime-ready experiment configuration.

    The model embeds validated input parameters, the job identifier assigned by
    the runner, and the root path for experiment artefacts.
    """

    input: InputConfig
    job_id: str  # (filename) UUID given from runner
    root_path: Path

    def create_rng(self) -> Generator:
        """Instantiate a NumPy RNG based on the configured seed.

        Returns:
            numpy.random.Generator: A default_rng instance seeded when
            ``input.experiment.seed`` is provided; otherwise a random
            seed is used.
        """
        seed = self.input.experiment.seed
        if seed is None:
            return default_rng()
        return default_rng(seed)

    def estimate_overweight_probability(self, total_item_weight: int) -> float:
        """Estimate the probability that a random solution exceeds max weight.

        The returned value is used as a Bernoulli parameter ``p`` when biasing the
        initial population towards overweight (infeasible) individuals.

        Args:
            total_item_weight: Sum of weights of all available items.

        Raises:
            ValueError: If ``total_item_weight`` is less than 1.

        Returns:
            A probability in the range [0.0, 1.0].
        """
        if total_item_weight < 1:
            raise ValueError("total_item_weight must be greater than 0.")

        probability = self.input.data.max_weight / total_item_weight
        return max(0.0, min(1.0, probability))

    def as_dict(self) -> dict:
        """Return the configuration as a plain dictionary."""
        return self.model_dump()

    @property
    def data_filename(self) -> str:
        return self.input.data.data_filename

    @property
    def max_weight(self) -> int:
        return self.input.data.max_weight

    @property
    def population_size(self) -> int:
        return self.input.population.size

    @property
    def generations(self) -> int:
        return self.input.population.generations

    @property
    def stream_batch_size(self) -> int:
        return self.input.population.stream_batch_size

    @property
    def commit_mode(self) -> SwapType:
        return self.input.population.commit_mode

    @property
    def selection_type(self) -> SelectionType:
        return self.input.selection.type

    @property
    def selection_pressure(self) -> float | None:
        return self.input.selection.selection_pressure

    @property
    def tournament_size(self) -> int | None:
        return self.input.selection.tournament_size

    @property
    def crossover_type(self) -> CrossoverType:
        return self.input.genetic_operators.crossover_type

    @property
    def crossover_probability(self) -> float:
        return self.input.genetic_operators.crossover_probability

    @property
    def mutation_probability(self) -> float:
        return self.input.genetic_operators.mutation_probability

    @property
    def penalty_multiplier(self) -> float:
        return self.input.genetic_operators.penalty_multiplier

    @property
    def strict_weight_constraints(self) -> bool:
        return self.input.genetic_operators.strict_weight_constraints

    @property
    def seed(self) -> int | None:
        return self.input.experiment.seed

    @property
    def experiment_identifier(self) -> str | None:
        return self.input.experiment.identifier

    @property
    def log_level(self) -> LogLevel:
        return self.input.experiment.log_level
