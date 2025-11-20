"""Defines fixtures and hooks for testing.

This module contains standard pytest fixtures used across various
test files in the 'tests/' directory, including mocked path resolution
and data generation utilities.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver


@pytest.fixture
def test_only_pathresolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> PathResolver:
    """
    Pytest fixture creating a spoofed PathResolver instance for testing.

    The PROJECT_ROOT of the PathResolver is mocked to point to the temporary
    directory provided by the 'tmp_path' fixture.

    Args:
        tmp_path (pathlib.Path): Pytest std fixture for creating a temp directory.
        monkeypatch (pytest.MonkeyPatch): Pytest std fixture for setting attributes.

    Returns:
        PathResolver: An initialized PathResolver instance, where PROJECT_ROOT
                      is set to the temporary directory.
    """
    monkeypatch.setattr(PathResolver, "PROJECT_ROOT", tmp_path)
    path_resolver = PathResolver()
    path_resolver.initialize(filename_constant="pytest_temp_file")
    assert tmp_path == path_resolver.PROJECT_ROOT
    return path_resolver


@pytest.fixture
def test_only_rng() -> np.random.Generator:
    """
    Pytest fixture creating a deterministic random number generator.

    Returns:
        numpy.random.Generator: A NumPy Random Number Generator initialized
                                with a fixed seed (1234).
    """
    rng = np.random.default_rng(seed=1234)
    return rng


@pytest.fixture
def root_path() -> Path:
    """
    Pytest fixture providing the root directory of the project.

    Returns:
        pathlib.Path: The Path object pointing to the project's root directory.
    """
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def temp_file(test_only_pathresolver: PathResolver) -> Path:
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
    return temp_dir / temp_filename


@pytest.fixture
def experiment_config_factory() -> Callable[..., ExperimentConfig]:
    """
    Factory fixture for creating fully initialized ExperimentConfig instances in tests.

    This fixture returns an inner function that builds ExperimentConfig objects with
    reasonable defaults. Individual tests can override any parameter using keyword
    arguments, while avoiding repeated boilerplate configuration code.

    Usage example:
        def test_something(experiment_config_factory):
            config = experiment_config_factory(
                population_size=10,
                generations=5,
                max_weight=100,
                selection_type="roulette",
                crossover_probability=0.8,
                mutation_probability=0.05,
                penalty_multiplier=10,
            )
            ...

    Returns:
        Callable[..., ExperimentConfig]: A factory function that accepts configuration
        parameters as keyword arguments and returns an ExperimentConfig instance.
    """

    def _factory(
        *,
        population_size: int,
        generations: int,
        max_weight: int,
        selection_type: str,
        crossover_type: str,
        crossover_probability: float,
        mutation_probability: float,
        penalty_multiplier: float,
        seed: int | None = 1234,
        exp_identifier: int = 0,
        log_level: str = "INFO",
        stream_batch: int | None = None,
        selection_pressure: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> ExperimentConfig:
        return ExperimentConfig(
            data_filename="temp_test_data.csv",
            population_size=population_size,
            generations=generations,
            max_weight=max_weight,
            seed=seed,
            selection_type=selection_type,
            crossover_type=crossover_type,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            penalty=penalty_multiplier,
            experiment_identifier=exp_identifier,
            log_level=log_level,
            stream_batch_size=stream_batch,
            rng=rng,
            selection_pressure=selection_pressure,
        )

    return _factory
