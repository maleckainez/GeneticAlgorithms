"""Integration tests for the ExperimentStorage facade."""

from pathlib import Path

import numpy as np
import pytest
from src.ga_core.storage import ExperimentStorage, PathResolver, data_paths


def test_from_root_prepares_layout(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    storage = ExperimentStorage.from_root(
        root=root, filename="demo", data_file_name="f_case"
    )

    assert storage.children_path() == root / "temp" / "child_demo.dat"
    assert storage.population_path() == root / "temp" / "demo.dat"
    assert (root / "temp").exists()
    assert (root / "output").exists()
    assert (root / "logs").exists()
    assert (root / "output" / "plots").exists()

    (root / "temp" / "artifact.txt").write_text("data")
    storage.remove_temp_data()
    assert not (root / "temp").exists()
    assert (root / "output").exists()


def test_population_memmap_round_trip(tmp_path: Path) -> None:
    storage = ExperimentStorage.from_root(
        root=tmp_path / "experiment", filename="demo", data_file_name="f_case"
    )

    storage.create_population_memmap(population_size=2, genome_length=3)
    memmap = storage.load_population_memmap(open_mode="r+")
    assert memmap.shape == (2, 3)
    assert memmap.dtype == np.uint8
    config_path = storage.population_path().with_suffix(".json")
    assert config_path.exists()
    memmap[1, 2] = 1
    memmap.flush()
    del memmap
    storage.remove_temp_data()
    assert not storage.population_path().exists()


def test_commit_children_swaps_population_file(tmp_path: Path) -> None:
    storage = ExperimentStorage.from_root(
        root=tmp_path / "experiment", filename="demo", data_file_name="f_case"
    )
    child = storage.children_path()
    payload = b"abcd"
    child.write_bytes(payload)

    storage.commit_children(expected_size=len(payload), retries=1)

    population = storage.population_path()
    assert population.exists()
    assert population.read_bytes() == payload
    assert not child.exists()


def test_data_paths_use_configured_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(data_paths, "SMALL_SCALE_DIR", tmp_path / "small")
    monkeypatch.setattr(data_paths, "SMALL_SCALE_OPTIMUM_DIR", tmp_path / "small-opt")

    storage = ExperimentStorage(
        layout=PathResolver(tmp_path / "experiment"),
        filename="demo",
        data_file_name="f_instance",
    )

    assert storage.data_dict_path() == tmp_path / "small" / "f_instance"
    assert storage.optimum_file() == tmp_path / "small-opt" / "f_instance"
