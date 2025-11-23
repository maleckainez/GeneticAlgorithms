"""Tests for filesystem path utilities used across the project."""

import os

import pytest
from src.classes.PathResolver import PathResolver


def test_get_dict_filepath_invalid_prefix_raises(
    test_only_pathresolver: PathResolver,
) -> None:
    with pytest.raises(Exception, match="Invalid file name"):
        test_only_pathresolver.get_dict_filepath("unknown_prefix_file")


def test_get_dict_filepath_small_scale(test_only_pathresolver: PathResolver) -> None:
    dict_path = test_only_pathresolver.get_dict_filepath("f1_l-d_kp_10_269")
    out_path = test_only_pathresolver.get_output_path()
    log_path = test_only_pathresolver.get_logging_path()
    assert dict_path is not None
    assert out_path is not None
    assert log_path is not None
    test_only_pathresolver.cleanup_temp_dir()
    assert os.path.exists(dict_path) is False


def test_get_dict_filepath_large_scale(test_only_pathresolver: PathResolver) -> None:
    dict_path = test_only_pathresolver.get_dict_filepath("knapPI_1_100_1000_1")
    out_path = test_only_pathresolver.get_output_path()
    log_path = test_only_pathresolver.get_logging_path()
    assert dict_path is not None
    assert out_path is not None
    assert log_path is not None
    test_only_pathresolver.cleanup_temp_dir()
    assert os.path.exists(dict_path) is False


def test_commit_children_missing_file_raises(
    test_only_pathresolver: PathResolver,
) -> None:
    expected_size = 10
    with pytest.raises(RuntimeError, match="Missing children file"):
        test_only_pathresolver.commit_children(expected_size=expected_size)


def test_commit_children_size_mismatch_raises(
    test_only_pathresolver: PathResolver,
) -> None:
    child_path = test_only_pathresolver.get_children_filepath()
    child_path.write_bytes(b"1")  # size 1
    expected_size = 5
    with pytest.raises(RuntimeError, match="Children size mismatch"):
        test_only_pathresolver.commit_children(expected_size=expected_size)


def test_commit_children_retries_and_raises_on_persistent_permission_error(
    test_only_pathresolver: PathResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_path = test_only_pathresolver.get_children_filepath()
    expected_size = 3
    child_path.write_bytes(b"123")

    monkeypatch.setattr(
        "src.classes.PathResolver.os.replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )
    monkeypatch.setattr(
        "src.classes.PathResolver.time.sleep", lambda *_args, **_kwargs: None
    )

    with pytest.raises(RuntimeError, match="Commit failed after 2 tries"):
        test_only_pathresolver.commit_children(expected_size=expected_size, retries=2)
    assert child_path.exists()


def test_commit_children_moves_file_when_size_matches(
    test_only_pathresolver: PathResolver,
) -> None:
    child_path = test_only_pathresolver.get_children_filepath()
    expected_size = 4
    child_path.write_bytes(b"1234")

    test_only_pathresolver.commit_children(expected_size=expected_size)

    population_path = test_only_pathresolver.get_temp_path() / (
        f"{test_only_pathresolver.filename_constant}.dat"
    )
    assert not child_path.exists()
    assert population_path.exists()
    assert population_path.stat().st_size == expected_size


def test_not_initialized_get_temp_path() -> None:
    path_resolver = PathResolver()
    with pytest.raises(RuntimeError, match="Directories were not initialized"):
        path_resolver.get_temp_path()
    with pytest.raises(RuntimeError, match="Directories were not initialized"):
        path_resolver.get_output_path()
    with pytest.raises(RuntimeError, match="Directories were not initialized"):
        path_resolver.get_logging_path()


def test_not_initialized_get_plot_path() -> None:
    path_resolver = PathResolver()
    with pytest.raises(RuntimeError, match="Directories were not initialized"):
        path_resolver.get_plot_path()


def test_get_plot_path_returns_initialized_path(
    test_only_pathresolver: PathResolver,
) -> None:
    plot_path = test_only_pathresolver.get_plot_path()
    assert plot_path.exists()


def test_get_optimum_path_small_and_large(test_only_pathresolver: PathResolver) -> None:
    test_only_pathresolver.filename_constant = "f123"
    small_optimum = test_only_pathresolver.get_optimum_path()
    assert "low-dimensional-optimum" in str(small_optimum)

    test_only_pathresolver.filename_constant = "knap123"
    large_optimum = test_only_pathresolver.get_optimum_path()
    assert "large_scale-optimum" in str(large_optimum)


def test_get_optimum_path_invalid_prefix_raises(
    test_only_pathresolver: PathResolver,
) -> None:
    test_only_pathresolver.filename_constant = "invalid"
    with pytest.raises(Exception, match="Invalid file name"):
        test_only_pathresolver.get_optimum_path()
