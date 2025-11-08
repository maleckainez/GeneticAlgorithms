import os
import numpy as np
from src.methods.utils import (
    create_population_file,
    find_temp_directory,
    load_memmap,
)
from pathlib import Path

SEED = 2137
rng = np.random.default_rng(SEED)


def test_create_population_basic():
    population_size = 2500
    genome_length = 100
    batch = 500
    create_population_file(population_size, genome_length, batch, rng=rng)
    population, config = load_memmap()
    pop_path = Path(find_temp_directory() / "population.dat")
    # Check does population have the right length (population_size)
    assert population.shape[0] == population_size
    # Check does population config matches the right values
    assert config["population_size"] == population_size
    assert config["genome_length"] == genome_length
    assert config["filesize"] == os.path.getsize(pop_path)
    # Check does all population lines have the right width (genome length)
    assert population.shape[1] == genome_length
    # Check does all values are binary
    assert np.all(np.isin(population, [0, 1])), "Non-binary values found"
    # Check dtype of batches
    assert population.dtype == np.uint8
    if os.path.exists(find_temp_directory() / "population.dat"):
        os.remove(pop_path)
        os.remove(find_temp_directory() / "population.json")


def test_create_population_big_width():
    population_size = 2500
    genome_length = int(1e6)
    batch = 500
    create_population_file(population_size, genome_length, batch, rng=rng)
    population, config = load_memmap()
    pop_path = Path(find_temp_directory() / "population.dat")
    # Check does population have the right length (population_size)
    assert population.shape[0] == population_size
    # Check does population config matches the right values
    assert config["population_size"] == population_size
    assert config["genome_length"] == genome_length
    assert config["filesize"] == os.path.getsize(pop_path)
    # Check does all population lines have the right width (genome length)
    assert population.shape[1] == genome_length
    # Check does all values are binary
    assert np.all(np.isin(population, [0, 1])), "Non-binary values found"
    # Check dtype of batches
    assert population.dtype == np.uint8
    if os.path.exists(find_temp_directory() / "population.dat"):
        os.remove(pop_path)
        os.remove(find_temp_directory() / "population.json")


def test_create_population_big_pop():
    population_size = int(1e6)
    genome_length = 100
    batch = 500
    create_population_file(population_size, genome_length, batch, rng=rng)
    population, config = load_memmap()
    pop_path = Path(find_temp_directory() / "population.dat")
    # Check does population have the right length (population_size)
    assert population.shape[0] == population_size
    # Check does population config matches the right values
    assert config["population_size"] == population_size
    assert config["genome_length"] == genome_length
    assert config["filesize"] == os.path.getsize(pop_path)
    # Check does all population lines have the right width (genome length)
    assert population.shape[1] == genome_length
    # Check does all values are binary
    assert np.all(np.isin(population, [0, 1])), "Non-binary values found"
    # Check dtype of batches
    assert population.dtype == np.uint8
    if os.path.exists(find_temp_directory() / "population.dat"):
        os.remove(pop_path)
        os.remove(find_temp_directory() / "population.json")
