"""Crossover and mutation helpers for the genetic algorithm."""

from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from src.classes.ChildrenHandler import ChildrenHandler
from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver
from src.classes.PopulationHandler import PopulationHandler

genome_array = NDArray[np.uint8]
mask_array = NDArray[np.bool_]
memmap_array = np.memmap[tuple[int, int], np.dtype[np.uint8]]
kernel_type = Callable[
    [genome_array, genome_array, genome_array, genome_array, mask_array],
    Tuple[genome_array, genome_array],
]


class Reproduction:
    """Performs crossover and mutation for a given parent pool."""

    def __init__(
        self,
        parent_pool: list[int],
        config: ExperimentConfig,
        paths: PathResolver,
    ) -> None:
        """Initialize reproduction with parent pool and configuration.

        Args:
            parent_pool (np.ndarray): Array of selected parents.
            config (ExperimentConfig): Experiment parameters and RNG.
            paths (PathResolver): Path helper for temporary files.
        """
        self.parent_pool = parent_pool
        self.config = config
        self.paths = paths
        self.parent_pairs: np.ndarray
        self.rng = self.config.rng
        self._pair_parents()

    def single_crossover(
        self, pop_manager: PopulationHandler, children_manager: ChildrenHandler
    ) -> None:
        """Run single-point crossover for the current parent pairs.

        Args:
            pop_manager (PopulationHandler): Population memmap handler.
            children_manager (ChildrenHandler): Children memmap handler.
        """
        self._calculation_runner(self._kernel_single, pop_manager, children_manager)

    def double_crossover(
        self, pop_manager: PopulationHandler, children_manager: ChildrenHandler
    ) -> None:
        """Run double-point crossover for the current parent pairs.

        Args:
            pop_manager (PopulationHandler): Population memmap handler.
            children_manager (ChildrenHandler): Children memmap handler.
        """
        self._calculation_runner(self._kernel_double, pop_manager, children_manager)

    def _pair_parents(self) -> None:
        """Shuffle parent pool into pairs."""
        assert self.rng is not None
        self.parent_pairs = self.rng.permutation(self.parent_pool).reshape(-1, 2)

    def _setup(self) -> None:
        """Load batch and probability settings from config."""
        self.stream_batch = self.config.stream_batch_size
        self.crossover_probability = self.config.crossover_probability
        self.mutation_probability = self.config.mutation_probability

    def _kernel_single(
        self,
        c1: genome_array,
        c2: genome_array,
        p1: genome_array,
        p2: genome_array,
        mask: mask_array,
    ) -> Tuple[genome_array, genome_array]:
        """Apply single crossover mask to two children."""
        assert self.rng is not None
        batch_size = int(c1.shape[0])
        genome_length = int(c1.shape[1])
        cut_columns: NDArray[np.int64] = self.rng.integers(
            1, genome_length, size=batch_size
        )
        column_index = np.arange(genome_length)

        cut_mask = column_index[None, :] >= cut_columns[:, None]
        cut_mask &= mask[:, None]
        c1[cut_mask] = p2[cut_mask]
        c2[cut_mask] = p1[cut_mask]
        return c1, c2

    def _kernel_double(
        self,
        c1: genome_array,
        c2: genome_array,
        p1: genome_array,
        p2: genome_array,
        mask: mask_array,
    ) -> Tuple[genome_array, genome_array]:
        """Apply double crossover mask to two children."""
        assert self.rng is not None
        batch_size = int(c1.shape[0])
        genome_length = int(c1.shape[1])
        start_cut_col: NDArray[np.int64] = self.rng.integers(
            1, genome_length - 1, size=batch_size
        )
        stop_cut_col = self.rng.integers(
            start_cut_col + 1, genome_length, size=batch_size
        )
        column_index = np.arange(genome_length)
        cut_mask = (column_index[None, :] >= start_cut_col[:, None]) & (
            column_index[None, :] < stop_cut_col[:, None]
        )
        cut_mask &= mask[:, None]
        c1[cut_mask] = p2[cut_mask]
        c2[cut_mask] = p1[cut_mask]
        return c1, c2

    def _calculation_runner(
        self,
        kernel: kernel_type,
        pop_manager: PopulationHandler,
        children_manager: ChildrenHandler,
    ) -> None:
        """Execute crossover and mutation in streamed batches.

        Args:
            kernel (kernel_type): Crossover kernel to apply.
            pop_manager (PopulationHandler): Population memmap handler.
            children_manager (ChildrenHandler): Children memmap handler.
        """
        population = pop_manager.get_pop_handle()
        children = children_manager.get_children_handle()
        assert population is not None
        assert children is not None
        self._setup()
        assert self.stream_batch is not None and self.rng is not None
        for start in range(0, len(self.parent_pairs), self.stream_batch):
            stop = min(start + self.stream_batch, len(self.parent_pairs))
            parent_indices = self.parent_pairs[start:stop]
            p1 = population[parent_indices[:, 0]]
            p2 = population[parent_indices[:, 1]]
            c1, c2 = p1.copy(), p2.copy()
            mask: mask_array = (
                self.rng.random(size=stop - start) < self.crossover_probability
            )
            c1, c2 = kernel(c1, c2, p1, p2, mask)
            if self.mutation_probability > 0:
                self._mutation(c1, c2)
            children[start * 2 : stop * 2] = np.concatenate((c1, c2), axis=0)
            children.flush()

    def _mutation(
        self, c1: genome_array, c2: genome_array
    ) -> Tuple[genome_array, genome_array]:
        """Mutate children genomes in place."""
        assert self.rng is not None
        mask1 = self.rng.random(size=c1.shape) < self.mutation_probability
        mask2 = self.rng.random(size=c2.shape) < self.mutation_probability
        c1[mask1] ^= 1
        c2[mask2] ^= 1
        return c1, c2
