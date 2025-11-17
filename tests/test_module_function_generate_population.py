from src.methods.utils import create_population_file
from pathlib import Path
import json
import numpy as np


def _validate_population_file(pr, rng, data):
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
    with open(json_file, "r") as f:
        config = json.load(f)
    filesize = config["filesize"]
    # assert that json and real filesize matches theoretical one
    assert filesize == data[0] * data[1]
    assert filesize == mmap_file.stat().st_size
    try:
        mmap = np.memmap(
            mmap_file,
            dtype=np.uint8,
            mode="r",
            shape=(config["population_size"], config["genome_length"]),
        )
        assert mmap.shape == (data[0], data[1])
    finally:
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
