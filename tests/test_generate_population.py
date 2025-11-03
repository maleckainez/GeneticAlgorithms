import json
import os
import numpy as np
from src.methods.utils import create_population_file
from src.methods.utils import load_data

SEED = 2137
rng = np.random.default_rng(SEED)


def test_create_population_basic():
    population_size = 2500
    genome_length = 100
    batch = 500
    create_population_file(population_size, genome_length, batch, rng=rng)
    population = np.memmap(
        "population.dat",
        dtype=np.uint8,
        mode="r",
        shape=(population_size, genome_length),
    )
    assert len(population) == population_size
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
        os.remove("population.json")


def test_create_population_big_width():
    population_size = 1000
    genome_length = int(1e6)
    batch = 500
    create_population_file(population_size, genome_length, batch, rng)
    population = np.memmap(
        "population.dat",
        dtype=np.uint8,
        mode="r",
        shape=(population_size, genome_length),
    )
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
        os.remove("population.json")


def test_create_population_big_pop():
    population_size = int(1e6)
    genome_length = 1000
    batch = 500
    create_population_file(population_size, genome_length, batch, rng)
    population = np.memmap(
        "population.dat",
        dtype=np.uint8,
        mode="r",
        shape=(population_size, genome_length),
    )
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
        os.remove("population.json")


def test_json_and_real_data():
    PATH = "../dane AG 2/large_scale/knapPI_1_100_1000_1"
    POPSIZE = 1000
    GENOME_LENGTH = len(load_data(PATH))
    create_population_file(1000, GENOME_LENGTH, 100, rng)
    with open("population.json", "r") as f:
        cfg = json.load(f)

    assert cfg["filesize"] == os.path.getsize(
        "population.dat"
    ), f"FILE CORRUPTED json size != file size"
    dtype = np.dtype(cfg["data_type"])
    population = np.memmap(
        cfg["filename"],
        dtype=dtype,
        mode="r",
        shape=(cfg["population_size"], cfg["genome_length"]),
    )
    # Check does the sum of the batches equals to population size
    total_rows = len(population)
    assert total_rows == POPSIZE, f"Expected {POPSIZE}, got {total_rows}"

    # Check does all population lines have the right width (genome length)
    assert population.shape[1] == GENOME_LENGTH

    # Check does all values are binary
    assert np.all(np.isin(population, [0, 1])), "Non-binary values found"

    # Check dtype of batches
    assert population.dtype == np.uint8

    if os.path.exists("population.dat"):
        os.remove("population.dat")
        os.remove("population.json")
