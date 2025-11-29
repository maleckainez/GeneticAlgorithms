"""Tests for population memmap helpers in the storage layer."""

import json
from pathlib import Path

import numpy as np
import pytest
from src.ga_core.storage import PathResolver
from src.ga_core.storage.directory_utils import ensure_layout_paths
from src.ga_core.storage.population_files import (
    children_filepath,
    commit_children,
    create_population_file,
    load_memmap,
    population_filepath,
)


def test_population_and_children_filepaths_include_temp_dir(tmp_path: Path) -> None:
    layout = PathResolver(tmp_path)
    ensure_layout_paths(layout)

    population = population_filepath(layout, "demo")
    children = children_filepath(layout, "demo")

    assert population.name == "demo.dat"
    assert children.name == "child_demo.dat"
    assert population.parent == layout.temp
    assert children.parent == layout.temp


def test_commit_children_validates_presence_and_size(tmp_path: Path) -> None:
    layout = PathResolver(tmp_path)
    ensure_layout_paths(layout)

    with pytest.raises(RuntimeError, match="Missing children file"):
        commit_children(layout=layout, file_name="demo", expected_size=4, retries=1)

    child = children_filepath(layout, "demo")
    child.write_bytes(b"abc")
    with pytest.raises(RuntimeError, match="Children size mismatch"):
        commit_children(layout=layout, file_name="demo", expected_size=4, retries=1)


def test_commit_children_replaces_population_file(tmp_path: Path) -> None:
    layout = PathResolver(tmp_path)
    ensure_layout_paths(layout)

    child = children_filepath(layout, "demo")
    payload = b"seed"
    child.write_bytes(payload)

    commit_children(layout=layout, file_name="demo", expected_size=len(payload))

    population = population_filepath(layout, "demo")
    assert population.exists()
    assert population.read_bytes() == payload
    assert not child.exists()


def test_create_and_load_memmap_round_trip(tmp_path: Path) -> None:
    layout = PathResolver(tmp_path)
    ensure_layout_paths(layout)

    create_population_file(
        population_size=3,
        genome_length=2,
        temp=layout.temp,
        filename="demo",
        data_type=np.uint8,
    )

    memmap, config = load_memmap(temp=layout.temp, filename="demo", open_mode="r+")
    assert memmap.shape == (3, 2)
    assert memmap.dtype == np.uint8
    assert config["population_size"] == 3
    assert config["genome_length"] == 2
    assert config["filesize"] == memmap.nbytes
    memmap[0, 0] = 1
    memmap.flush()
    del memmap


def test_load_memmap_reports_missing_or_corrupted_files(tmp_path: Path) -> None:
    layout = PathResolver(tmp_path)
    ensure_layout_paths(layout)

    with pytest.raises(FileNotFoundError):
        load_memmap(temp=layout.temp, filename="missing")

    corrupted = layout.temp / "corrupted.json"
    corrupted.touch()
    with pytest.raises(ValueError, match="corrupted"):
        load_memmap(temp=layout.temp, filename="corrupted")

    dat_path = layout.temp / "broken.dat"
    dat_path.write_bytes(b"\x00" * 8)
    bad_config = {
        "filename": str(dat_path),
        "data_type": "uint8",
        "population_size": 2,
        "genome_length": 4,
        "filesize": 4,
    }
    (layout.temp / "broken.json").write_text(json.dumps(bad_config))
    with pytest.raises(ValueError, match="corrupted"):
        load_memmap(temp=layout.temp, filename="broken")
