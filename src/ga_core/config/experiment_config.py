"""Typed experiment configuration and helpers.

This module provides a thin wrapper around :class:`InputConfig` that adds runtime
utilities needed by runners, such as RNG creation and probability estimation for
overweight genomes.
"""

from pathlib import Path

from numpy.random import Generator, default_rng
from pydantic import BaseModel

from .input_config_scheme import InputConfig


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
