"""Batch crossover kernels for binary-encoded genetic algorithms.

Provides vectorized crossover operations that process entire batches
of parent pairs simultaneously for memory efficiency.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

gen_arr = NDArray[np.uint8]
mask_arr = NDArray[np.bool_]


def single_point_crossover(
    c1: gen_arr,
    c2: gen_arr,
    p1: gen_arr,
    p2: gen_arr,
    mask: mask_arr,
    rng: np.random.Generator,
) -> Tuple[gen_arr, gen_arr]:
    """Perform single-point crossover on a batch of parent pairs.

    Randomly selects a single cut point for each pair and swaps genetic material
    between parents. Operates in-place on children arrays.

    Args:
        c1: Genome array of first children (shape: [batch_size, genome_length]).
        c2: Genome array of second children (shape: [batch_size, genome_length]).
        p1: Genome array of first parents (shape: [batch_size, genome_length]).
        p2: Genome array of second parents (shape: [batch_size, genome_length]).
        mask: Boolean array indicating which pairs should undergo crossover.
        rng: NumPy random generator for reproducible cut point selection.

    Returns:
        Tuple of modified children genome arrays (c1, c2).
    """
    batch_size = int(c1.shape[0])
    genome_length = int(c1.shape[1])
    cut_columns: NDArray[np.int64] = rng.integers(1, genome_length, size=batch_size)
    column_index = np.arange(genome_length)

    cut_mask = column_index[None, :] >= cut_columns[:, None]
    cut_mask &= mask[:, None]
    c1[cut_mask] = p2[cut_mask]
    c2[cut_mask] = p1[cut_mask]
    return c1, c2


def double_point_crossover(
    c1: gen_arr,
    c2: gen_arr,
    p1: gen_arr,
    p2: gen_arr,
    mask: mask_arr,
    rng: np.random.Generator,
) -> Tuple[gen_arr, gen_arr]:
    """Perform two-point crossover on a batch of parent pairs.

    Randomly selects two cut points for each pair and swaps the genetic material
    between them. Operates in-place on children arrays.

    Args:
        c1: Genome array of first children (shape: [batch_size, genome_length]).
        c2: Genome array of second children (shape: [batch_size, genome_length]).
        p1: Genome array of first parents (shape: [batch_size, genome_length]).
        p2: Genome array of second parents (shape: [batch_size, genome_length]).
        mask: Boolean array indicating which pairs should undergo crossover.
        rng: NumPy random generator for reproducible cut point selection.

    Returns:
        Tuple of modified children genome arrays (c1, c2).
    """
    batch_size = int(c1.shape[0])
    genome_length = int(c1.shape[1])
    start_cut_col: NDArray[np.int64] = rng.integers(
        1, genome_length - 1, size=batch_size
    )
    stop_cut_col = rng.integers(start_cut_col + 1, genome_length, size=batch_size)
    column_index = np.arange(genome_length)
    cut_mask = (column_index[None, :] >= start_cut_col[:, None]) & (
        column_index[None, :] < stop_cut_col[:, None]
    )
    cut_mask &= mask[:, None]
    c1[cut_mask] = p2[cut_mask]
    c2[cut_mask] = p1[cut_mask]
    return c1, c2
