import os

import numpy as np
from methods.utils import create_population_file
from methods.utils import load_data

SEED = 2137


def test_create_population_basic():
    population_size = 2500
    genome_length = 100
    batch = 500
    population = create_population_file(population_size, genome_length, batch, SEED)
    # Check does the sum of the batches equals to population size
    total_rows = len(population)
    assert (
        total_rows == population_size
    ), f"Expected {population_size}, got {total_rows}"

    # Check does all population lines have the right width (genome length)
    assert population.shape[1] == genome_length

    # Check does all values are binary
    assert np.all(np.isin(population, [0, 1])), "Non-binary values found"

    # Check dtype of batches
    assert population.dtype == np.uint8

    if os.path.exists("population.dat"):
        os.remove("population.dat")


def test_create_population_big_width():
    population_size = 1000
    genome_length = int(1e6)
    batch = 500
    population = create_population_file(population_size, genome_length, batch, SEED)
    # Check does the sum of the batches equals to population size
    total_rows = len(population)
    assert (
        total_rows == population_size
    ), f"Expected {population_size}, got {total_rows}"

    # Check does all population lines have the right width (genome length)
    assert population.shape[1] == genome_length

    # Check does all values are binary
    assert np.all(np.isin(population, [0, 1])), "Non-binary values found"

    # Check dtype of batches
    assert population.dtype == np.uint8

    if os.path.exists("population.dat"):
        os.remove("population.dat")


def test_create_population_big_pop():
    population_size = int(1e6)
    genome_length = 1000
    batch = 500
    population = create_population_file(population_size, genome_length, batch, SEED)
    # Check does the sum of the batches equals to population size
    total_rows = len(population)
    assert (
        total_rows == population_size
    ), f"Expected {population_size}, got {total_rows}"

    # Check does all population lines have the right width (genome length)
    assert population.shape[1] == genome_length

    # Check does all values are binary
    assert np.all(np.isin(population, [0, 1])), "Non-binary values found"

    # Check dtype of batches
    assert population.dtype == np.uint8

    if os.path.exists("population.dat"):
        os.remove("population.dat")
