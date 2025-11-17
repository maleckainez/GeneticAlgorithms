import pytest
import numpy as np

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
def root_path():
    from pathlib import Path
    root_path = Path(__file__).resolve().parents[1]
    return root_path