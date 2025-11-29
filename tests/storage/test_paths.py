"""Tests for storage path resolution and directory helpers.

These tests cover PathResolver behavior and utilities that prepare or clean up
experiment directories.
"""

from pathlib import Path

import pytest
from src.ga_core.storage import PathResolver, data_paths
from src.ga_core.storage.directory_utils import cleanup_temp, ensure_layout_paths


def test_path_resolver_defines_layout_without_creating_dirs(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    resolver = PathResolver(root)

    assert resolver.temp == root / "temp"
    assert resolver.output == root / "output"
    assert resolver.logs == root / "logs"
    assert resolver.plots == root / "output" / "plots"
    assert not resolver.temp.exists()
    assert not resolver.output.exists()
    assert not resolver.logs.exists()
    assert not resolver.plots.exists()


def test_ensure_layout_paths_creates_directories_and_cleanup_removes_temp(
    tmp_path: Path,
) -> None:
    layout = PathResolver(tmp_path / "run")
    ensure_layout_paths(layout)

    for path in (layout.temp, layout.output, layout.logs, layout.plots):
        assert path.exists()

    artifact = layout.temp / "artifact.bin"
    artifact.write_bytes(b"data")
    cleanup_temp(layout)

    assert not layout.temp.exists()
    assert layout.output.exists()
    assert layout.logs.exists()
    assert layout.plots.parent.exists()
    cleanup_temp(layout)
    assert not layout.temp.exists()


def test_data_paths_resolve_by_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(data_paths, "SMALL_SCALE_DIR", tmp_path / "small")
    monkeypatch.setattr(data_paths, "LARGE_SCALE_DIR", tmp_path / "large")

    small = data_paths.resolve_data_dict_path("f_instance")
    large = data_paths.resolve_data_dict_path("knap_instance")

    assert small == tmp_path / "small" / "f_instance"
    assert large == tmp_path / "large" / "knap_instance"


def test_optimum_paths_match_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(data_paths, "SMALL_SCALE_OPTIMUM_DIR", tmp_path / "small-opt")
    monkeypatch.setattr(data_paths, "LARGE_SCALE_OPTIMUM_DIR", tmp_path / "large-opt")

    small_opt = data_paths.resolve_optimum_file("f_instance")
    large_opt = data_paths.resolve_optimum_file("knap_instance")

    assert small_opt == tmp_path / "small-opt" / "f_instance"
    assert large_opt == tmp_path / "large-opt" / "knap_instance"


def test_data_path_resolution_rejects_invalid_prefix() -> None:
    with pytest.raises(ValueError):
        data_paths.resolve_data_dict_path("x_invalid")
    with pytest.raises(ValueError):
        data_paths.resolve_optimum_file("x_invalid")
