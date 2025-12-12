"""Integration tests for the ExperimentStorage facade."""

from pathlib import Path

import pytest
from src.ga_core.engine.population.directory_manager import DirectoryManager
from src.ga_core.storage import ExperimentStorage, data_paths
from src.ga_core.storage.directory_utils import ensure_layout_paths


def test_from_root_prepares_layout(tmp_path: Path) -> None:
    """Test that ExperimentStorage creates layout directories."""
    root = tmp_path / "experiment"
    layout = DirectoryManager(root=root)
    ensure_layout_paths(layout)

    storage = ExperimentStorage(layout=layout, job_id="demo", data_file_name="f_case")

    assert storage.children_name() == "demo_2.dat"
    assert storage.population_name() == "demo.dat"
    assert (root / "temp").exists()
    assert (root / "output").exists()
    assert (root / "logs").exists()
    assert (root / "output" / "plots").exists()

    (root / "temp" / "artifact.txt").write_text("data")
    storage.remove_temp_data()
    assert not (root / "temp").exists()
    assert (root / "output").exists()


def test_data_paths_use_configured_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that data_dict_path and optimum_file use configured directories."""
    monkeypatch.setattr(data_paths, "SMALL_SCALE_DIR", tmp_path / "small")
    monkeypatch.setattr(data_paths, "SMALL_SCALE_OPTIMUM_DIR", tmp_path / "small-opt")

    layout = DirectoryManager(root=tmp_path / "experiment")
    storage = ExperimentStorage(
        layout=layout,
        job_id="demo",
        data_file_name="f_instance",
    )

    assert storage.data_dict_path() == tmp_path / "small" / "f_instance"
    assert storage.optimum_file() == tmp_path / "small-opt" / "f_instance"
