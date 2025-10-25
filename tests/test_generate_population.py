import numpy as np
from methods.utils import create_population_stream


def test_create_population_stream_basic():
    population_size = 2500
    genome_length = 100
    stream_batch = 500
    batches = list(
        create_population_stream(population_size, genome_length, stream_batch)
    )
    # Check does the sum of the batches equals to population size
    total_rows = sum(batch.shape[0] for batch in batches)
    assert (
        total_rows == population_size
    ), f"Expected {population_size}, got {total_rows}"

    # Check does all batches have the right width (genome length)
    assert all(batch.shape[1] == genome_length for batch in batches)

    # Check does all values are binary
    for batch in batches:
        assert np.all(np.isin(batch, [0, 1])), "Non-binary values found"

    # Check dtype of batches
    assert all(batch.dtype == np.uint8 for batch in batches)

def test_create_population_stream_big():
    population_size = 100000
    genome_length = 10000
    stream_batch = 500
    batches = list(
        create_population_stream(population_size, genome_length, stream_batch)
    )
    # Check does the sum of the batches equals to population size
    total_rows = sum(batch.shape[0] for batch in batches)
    assert (
        total_rows == population_size
    ), f"Expected {population_size}, got {total_rows}"

    # Check does all batches have the right width (genome length)
    assert all(batch.shape[1] == genome_length for batch in batches)

    # Check does all values are binary
    for batch in batches:
        assert np.all(np.isin(batch, [0, 1])), "Non-binary values found"

    # Check dtype of batches
    assert all(batch.dtype == np.uint8 for batch in batches)

