"""Pure generators for initial population batches."""

from typing import Iterator

import numpy as np


def generate_initial_batches(
    population_size: int,
    genome_length: int,
    stream_batch_size: int,
    rng: np.random.Generator,
    overweight_probability: float,
) -> Iterator[np.ndarray]:
    """Yield binary batches for the initial population.

    Args:
        population_size: Total number of individuals to generate.
        genome_length: Number of genes per individual.
        stream_batch_size: Size of each yielded batch.
        rng: Random generator used for reproducible sampling.
        overweight_probability: Probability of gene set to 1.
    """
    if overweight_probability < 0 or overweight_probability > 1:
        raise ValueError("Overweight probability is out of bounds")
    p = overweight_probability
    if stream_batch_size < 1 or stream_batch_size > 10000:
        raise ValueError("Stream batch size is out of bounds")
    for start in range(0, population_size, stream_batch_size):
        stop = min(start + stream_batch_size, population_size)
        batch = (rng.random(size=(stop - start, genome_length)) < p).astype(np.uint8)
        yield batch
