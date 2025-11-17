import pytest
from pathlib import Path
import numpy as np
import json


@pytest.fixture
def test_only_pathresolver(tmp_path, monkeypatch):
    from src.classes.PathResolver import PathResolver as PR
    monkeypatch.setattr(PR, "PROJECT_ROOT", tmp_path)
    path_resolver = PR()
    path_resolver.initialize(filename_constant="pytest_temp_file")
    assert tmp_path == path_resolver.PROJECT_ROOT
    return path_resolver

@pytest.fixture
def test_only_rng():
    rng = np.random.default_rng(seed=1234)
    return rng

@pytest.fixture
def validate_population_file(test_only_pathresolver, test_only_rng, population):
    pr = test_only_pathresolver
    temp_path = pr.get_temp_path()
    name = pr.filename_constant
    assert temp_path.exists()
    assert "pytest_temp_file" in str(temp_path)
    population(temp_path, test_only_rng, name)
    mmap_file = Path(temp_path/f"{name}.dat")
    json_file = Path(temp_path/f"{name}.json")
    assert mmap_file.exists()
    assert json_file.exists()
    assert mmap_file.stat().st_size != 0
    assert json_file.stat().st_size != 0
    with open(json_file, "r") as f:
        config = json.load(f)
    filesize = config["filesize"]
    assert filesize == mmap_file.stat().st_size