# pylint: disable=missing-function-docstring
"""Defines tests for create_population_file function

This module contiains test specific only for testing function
located in src.methods.utils used to create starting populations
for GA algorithms. Tests below ensure that populations are created
correctly and i/o problems won't occur during memmap creation.
"""
from pathlib import Path
import json
import numpy as np
from src.methods.utils import create_population_file


def _validate_population_file(pr, rng, data):
    """Private function containing assertions to minimize code repetition.

    This function calls the tested function and runs multiple assertions using
    variables provided by callers (tests).

    Args:
        pr (src.classes.PathResolver): An initialized PathResolver instance.
        rng (np.random.RandomState): An initialized random number generator
            state with a predefined seed.
        data (list[int]): List containing integer values in the form
            [population_size, genome_length, stream_batch].
    """
    # define variables
    temp_path = pr.get_temp_path()
    name = pr.filename_constant
    mmap_file = Path(temp_path / f"{name}.dat")
    json_file = Path(temp_path / f"{name}.json")
    # assert that PathResolver fixture worked properly
    assert temp_path.exists()
    assert "pytest_temp_file" in str(temp_path)
    # create population using values from list
    create_population_file(
        population_size=data[0],
        genome_length=data[1],
        stream_batch=data[2],
        rng=rng,
        temp=temp_path,
        filename_constant=name,
    )
    # assert that datafile and json were created
    assert mmap_file.exists()
    assert json_file.exists()
    # assert that files are not corrupted (their weight is not null)
    assert mmap_file.stat().st_size != 0
    assert json_file.stat().st_size != 0
    # load json config file
    with open(json_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    filesize = config["filesize"]
    # assert that json and real filesize matches theoretical one
    assert filesize == data[0] * data[1]
    assert filesize == mmap_file.stat().st_size
    mmap = np.memmap(
        mmap_file,
        dtype=np.uint8,
        mode="r",
        shape=(config["population_size"], config["genome_length"]),
    )
    assert mmap.shape == (data[0], data[1])
    del mmap


def test_batch_equal_to_len(test_only_pathresolver, test_only_rng):
    _validate_population_file(
        pr=test_only_pathresolver, rng=test_only_rng, data=[10, 10, 10]
    )


def test_batch_smaller_to_len(test_only_pathresolver, test_only_rng):
    _validate_population_file(
        pr=test_only_pathresolver, rng=test_only_rng, data=[100, 10, 10]
    )


def test_batch_bigger_to_len(test_only_pathresolver, test_only_rng):
    _validate_population_file(
        pr=test_only_pathresolver, rng=test_only_rng, data=[100, 10, 500]
    )


def test_create_wide_pop(test_only_pathresolver, test_only_rng):
    _validate_population_file(
        pr=test_only_pathresolver, rng=test_only_rng, data=[10, 10000, 10]
    )


def test_create_long_pop(test_only_pathresolver, test_only_rng):
    _validate_population_file(
        pr=test_only_pathresolver, rng=test_only_rng, data=[10000, 100, 500]
    )
