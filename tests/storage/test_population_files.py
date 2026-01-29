"""Tests for population memmap helpers in the storage layer."""

import json
from pathlib import Path

import pytest
from src.ga_core.storage.directory_utils import ensure_layout_paths
from src.ga_core.storage.layout import StorageLayout
from src.ga_core.storage.population_files import (
    commit_children,
    create_empty_memmap,
    load_memmap,
)

create_population_file = create_empty_memmap


class _SimpleLayout(StorageLayout):
    """Simple test layout implementation."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._temp = root / "temp"
        self._output = root / "output"
        self._logs = root / "logs"
        self._plots = root / "output" / "plots"

    @property
    def temp(self) -> Path:
        """Return temp directory."""
        return self._temp

    @property
    def output(self) -> Path:
        """Return output directory."""
        return self._output

    @property
    def logs(self) -> Path:
        """Return logs directory."""
        return self._logs

    @property
    def plots(self) -> Path:
        """Return plots directory."""
        return self._plots


def test_population_and_children_filepaths_include_temp_dir(tmp_path: Path) -> None:
    """Test that population and children files are created in temp directory."""
    layout = _SimpleLayout(tmp_path)
    ensure_layout_paths(layout)

    population = layout.temp / "demo.dat"
    children = layout.temp / "child_demo.dat"

    assert population.name == "demo.dat"
    assert children.name == "child_demo.dat"
    assert population.parent == layout.temp
    assert children.parent == layout.temp


def test_commit_children_validates_presence_and_size(tmp_path: Path) -> None:
    """Test that commit_children validates file presence and size."""
    layout = _SimpleLayout(tmp_path)
    ensure_layout_paths(layout)

    with pytest.raises(RuntimeError, match="Missing children file"):
        commit_children(
            temp_path=layout.temp, file_name="demo", expected_size=4, retries=1
        )

    child = layout.temp / "demo_2.dat"
    child.write_bytes(b"abc")
    with pytest.raises(RuntimeError, match="Children size mismatch"):
        commit_children(
            temp_path=layout.temp, file_name="demo", expected_size=4, retries=1
        )


def test_commit_children_replaces_population_file(tmp_path: Path) -> None:
    """Test that commit_children atomically swaps children into population."""
    layout = _SimpleLayout(tmp_path)
    ensure_layout_paths(layout)

    child = layout.temp / "demo_2.dat"
    payload = b"seed"
    child.write_bytes(payload)

    commit_children(temp_path=layout.temp, file_name="demo", expected_size=len(payload))

    population = layout.temp / "demo.dat"
    assert population.exists()
    assert population.read_bytes() == payload
    assert not child.exists()


def test_load_memmap_reports_missing_or_corrupted_files(tmp_path: Path) -> None:
    """Test that load_memmap reports missing or corrupted files."""
    layout = _SimpleLayout(tmp_path)
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
