"""Defines fixtures and hooks for testing.

This module contains standard pytest fixtures used across various test files
in the 'tests/' directory, including mocked path resolution and data generation utilities.
"""

from pathlib import Path
import pytest
import numpy as np


@pytest.fixture
def test_only_pathresolver(tmp_path, monkeypatch):
    """
    Pytest fixture creating a spoofed PathResolver instance for testing.

    The PROJECT_ROOT of the PathResolver is mocked to point to the temporary
    directory provided by the 'tmp_path' fixture.

    Args:
        tmp_path (pathlib.Path): Pytest standard fixture for creating a temporary directory.
        monkeypatch (pytest.MonkeyPatch): Pytest standard fixture for setting attributes.

    Returns:
        PathResolver: An initialized PathResolver instance, where PROJECT_ROOT
                      is set to the temporary directory.
    """
    from src.classes.PathResolver import PathResolver as PR

    monkeypatch.setattr(PR, "PROJECT_ROOT", tmp_path)
    path_resolver = PR()
    path_resolver.initialize(filename_constant="pytest_temp_file")
    assert tmp_path == path_resolver.PROJECT_ROOT
    return path_resolver


@pytest.fixture
def test_only_rng():
    """
    Pytest fixture creating a deterministic random number generator.

    Returns:
        numpy.random.Generator: A NumPy Random Number Generator initialized
                                with a fixed seed (1234).
    """
    rng = np.random.default_rng(seed=1234)
    return rng


@pytest.fixture
def root_path():
    """
    Pytest fixture providing the root directory of the project.

    Returns:
        pathlib.Path: The Path object pointing to the project's root directory.
    """
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def temp_file(test_only_pathresolver):
    """
    Pytest fixture providing the Path object for a temporary output file.

    The file path is generated using the mocked PathResolver instance.

    Args:
        test_only_pathresolver (PathResolver): Custom fixture providing
                                               the mocked path resolver instance.

    Returns:
        pathlib.Path: The full Path object for the temporary output file.
    """
    temp_dir = test_only_pathresolver.get_temp_path()
    temp_filename = test_only_pathresolver.filename_constant
    return Path(temp_dir / temp_filename)
