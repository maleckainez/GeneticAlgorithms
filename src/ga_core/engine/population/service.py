"""Helpers to build and load population buffers using storage adapters."""

from __future__ import annotations

import numpy as np

from src.ga_core.engine.memory_policy import policy
from src.ga_core.storage.experiment_storage import ExperimentStorage
from src.ga_core.storage.population_files import create_empty_array

from .initialize import generate_initial_batches
from .types import PopulationType


def init_binary_population(
    population_size: int,
    genome_length: int,
    stream_batch_size: int,
    storage: ExperimentStorage,
    rng: np.random.Generator,
    overweight_probability: float,
) -> PopulationType:
    """Create initial population in RAM or memmap based on memory policy."""
    if policy.should_use_array_selection(
        population_size=population_size,
        genome_length=genome_length,
    ):
        return _binary_array(
            population_size=population_size,
            genome_length=genome_length,
            stream_batch_size=stream_batch_size,
            rng=rng,
            overweight_probability=overweight_probability,
        )

    return _binary_memmap(
        population_size=population_size,
        genome_length=genome_length,
        stream_batch_size=stream_batch_size,
        storage=storage,
        rng=rng,
        overweight_probability=overweight_probability,
    )


def initialize_empty_binary_children(
    population_size: int,
    genome_length: int,
    storage: ExperimentStorage,
) -> PopulationType:
    """Prepare empty children buffer (RAM or memmap) using storage adapter."""
    if policy.should_use_array_selection(
        population_size=population_size,
        genome_length=genome_length,
    ):
        return create_empty_array(population_size, genome_length)
    storage.create_empty_memmap(population_size, genome_length)
    return storage.load_population_memmap(open_mode="r+")


def _binary_array(
    population_size: int,
    genome_length: int,
    stream_batch_size: int,
    rng: np.random.Generator,
    overweight_probability: float,
) -> np.ndarray:
    initial_population = create_empty_array(population_size, genome_length)
    _fill_initial_population(
        initial_population=initial_population,
        population_size=population_size,
        genome_length=genome_length,
        stream_batch_size=stream_batch_size,
        rng=rng,
        overweight_probability=overweight_probability,
    )
    return initial_population


def _binary_memmap(
    population_size: int,
    genome_length: int,
    stream_batch_size: int,
    storage: ExperimentStorage,
    rng: np.random.Generator,
    overweight_probability: float,
) -> np.memmap:
    storage.create_empty_memmap(
        population_size=population_size,
        genome_length=genome_length,
    )
    initial_population = storage.load_population_memmap(open_mode="w+")
    _fill_initial_population(
        initial_population=initial_population,
        population_size=population_size,
        genome_length=genome_length,
        stream_batch_size=stream_batch_size,
        rng=rng,
        overweight_probability=overweight_probability,
    )
    initial_population.flush()
    return initial_population


def _fill_initial_population(
    initial_population: PopulationType,
    population_size: int,
    genome_length: int,
    stream_batch_size: int,
    rng: np.random.Generator,
    overweight_probability: float,
) -> None:
    offset = 0
    for batch in generate_initial_batches(
        population_size=population_size,
        genome_length=genome_length,
        stream_batch_size=stream_batch_size,
        rng=rng,
        overweight_probability=overweight_probability,
    ):
        next_offset = offset + len(batch)
        initial_population[offset:next_offset] = batch
        offset = next_offset
