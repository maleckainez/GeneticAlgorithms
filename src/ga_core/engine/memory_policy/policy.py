"""Heuristics for selecting in-memory vs memmap storage for populations."""

import numpy as np
import psutil


def should_use_array_selection(
    population_size: int,
    genome_length: int,
    dtype: type = np.uint8,
    ram_usage_fraction: float = 0.2,
    hard_max_ram_size: int | None = None,
) -> bool:
    """Return True when population fits comfortably in RAM, otherwise False.

    Args:
        population_size: Number of individuals in the population.
        genome_length: Number of genes per individual.
        dtype: NumPy dtype used for storage.
        ram_usage_fraction: Fraction of available RAM allowed for population data.
        hard_max_ram_size: Optional absolute byte limit for in-RAM arrays.
    """
    population_bytesize = population_size * genome_length * np.dtype(dtype).itemsize

    if hard_max_ram_size is not None:
        return population_bytesize < hard_max_ram_size

    avaliable_ram_size = psutil.virtual_memory().available
    return population_bytesize < ram_usage_fraction * avaliable_ram_size
