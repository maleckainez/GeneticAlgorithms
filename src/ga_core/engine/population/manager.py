"""Population lifecycle orchestrator using storage adapters."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.ga_core.config.input_config_scheme import SwapType
from src.ga_core.storage.experiment_storage import ExperimentStorage

from .service import init_binary_population, initialize_empty_binary_children
from .types import PopulationType


class PopulationManager:
    """Manage population and children buffers, including commits."""

    def __init__(
        self,
        population_size: int,
        genome_length: int,
        stream_batch_size: int,
        storage: ExperimentStorage,
        rng: np.random.Generator,
        overweight_probability: float,
        commit_mode: SwapType = SwapType.FLIP,
    ) -> None:
        """Initialize manager state and configuration.

        Args:
            population_size: Number of individuals in the population.
            genome_length: Number of genes per individual.
            stream_batch_size: Batch size used for initialization.
            storage: Storage adapter handling memmap creation/loading.
            rng: Random generator shared across the run.
            overweight_probability: Probability of gene set to 1.
            commit_mode: Swap strategy (flip in-memory or atomic on disk).
        """
        self._population: Optional[PopulationType] = None
        self._children: Optional[PopulationType] = None
        self._swap_mode: SwapType = commit_mode
        self._pop_size = population_size
        self._gen_len = genome_length
        self._batch = stream_batch_size
        self._storage = storage
        self._rng = rng
        self._q = overweight_probability

    @property
    def population(self) -> PopulationType:
        """Return current population array.

        Raises:
            RuntimeError: when population array was not initialized by manager.

        Returns:
            PopulationType: array or mmemap handle of population binary file.
        """
        if self._population is None:
            raise RuntimeError("Population not initialized")
        return self._population

    @property
    def children(self) -> PopulationType:
        """Return current children array.

        Raises:
            RuntimeError: when children array was not initialized by manager.

        Returns:
            PopulationType: array or mmemap handle of children binary file.
        """
        if self._children is None:
            raise RuntimeError("Children not initialized")
        return self._children

    def initialize_population(self) -> None:
        """Create initial population buffer (RAM or memmap)."""
        if self._population is not None:
            raise RuntimeError("Population already initialized!")
        self._population = init_binary_population(
            population_size=self._pop_size,
            genome_length=self._gen_len,
            stream_batch_size=self._batch,
            storage=self._storage,
            rng=self._rng,
            overweight_probability=self._q,
        )

    def initialize_children(self) -> None:
        """Create empty children buffer (RAM or memmap)."""
        if self._children is not None:
            raise RuntimeError("Children already initialized!")
        self._children = initialize_empty_binary_children(
            population_size=self._pop_size,
            genome_length=self._gen_len,
            storage=self._storage,
        )

    def init_pop_and_children(self) -> None:
        """Initialize both population and children buffers."""
        self.initialize_population()
        self.initialize_children()

    def commit_reproduction_of_population(self) -> None:
        """Persist or swap children into population according to commit mode."""
        if self._population is None or self._children is None:
            raise RuntimeError("Population or children not initialized")
        if self._swap_mode == SwapType.FLIP:
            self._commit_flip()
        elif self._swap_mode == SwapType.ATOMIC:
            self._commit_atomic()
        else:
            raise ValueError(
                f"{self._swap_mode} is not a valid file handling technique!"
            )

    def _commit_flip(self) -> None:
        """Swap in-memory buffers between population and children."""
        self._population, self._children = self._children, self._population

    def _commit_atomic(self) -> None:
        """Commit children to disk atomically and refresh buffers."""
        expected_size = self._pop_size * self._gen_len * np.dtype(np.uint8).itemsize
        # atomic swap on disk
        self._storage.commit_children(expected_size=expected_size)
        # reload population and recreate children buffer
        self._population = self._storage.load_population_memmap(open_mode="r+")
        self._storage.create_empty_children_memmap(
            population_size=self._pop_size,
            genome_length=self._gen_len,
        )
        self._children = self._storage.load_population_memmap(open_mode="r+")
