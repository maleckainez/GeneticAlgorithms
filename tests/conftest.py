"""Shared fixtures for legacy tests and refactored modules."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import numpy as np
import pytest
from src.ga_core.config import ExperimentConfig as CoreExperimentConfig
from src.ga_core.config import InputConfig
from src.ga_core.storage import ExperimentStorage
from src.ga_core.storage import PathResolver as CorePathResolver
from src.ga_core.storage.data_paths import resolve_data_dict_path, resolve_optimum_file
from src.ga_core.storage.directory_utils import cleanup_temp, ensure_layout_paths
from src.ga_core.storage.population_files import (
    children_filepath,
    commit_children,
)


class CompatPathResolver(CorePathResolver):
    """Provide the legacy PathResolver interface backed by the storage API."""

    PROJECT_ROOT = Path.cwd()

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            root = self.PROJECT_ROOT
        super().__init__(root)
        self.PROJECT_ROOT = root
        self.filename_constant: Optional[str] = None

    def initialize(self, filename_constant: str) -> None:
        self.filename_constant = filename_constant
        ensure_layout_paths(self)

    def _ensure_initialized(self) -> None:
        if self.filename_constant is None:
            raise RuntimeError("Directories were not initialized")

    def get_temp_path(self) -> Path:
        self._ensure_initialized()
        ensure_layout_paths(self)
        return self.temp

    def get_output_path(self) -> Path:
        self._ensure_initialized()
        ensure_layout_paths(self)
        return self.output

    def get_logging_path(self) -> Path:
        self._ensure_initialized()
        ensure_layout_paths(self)
        return self.logs

    def get_plot_path(self) -> Path:
        self._ensure_initialized()
        ensure_layout_paths(self)
        self.plots.mkdir(parents=True, exist_ok=True)
        return self.plots

    def get_children_filepath(self) -> Path:
        self._ensure_initialized()
        ensure_layout_paths(self)
        assert self.filename_constant is not None
        return children_filepath(self, self.filename_constant)

    def get_dict_filepath(self, data_file_name: str) -> Path:
        self._ensure_initialized()
        ensure_layout_paths(self)
        target = self.temp / data_file_name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = resolve_data_dict_path(data_file_name)
        if source.exists():
            target.write_bytes(source.read_bytes())
        else:
            target.touch()
        return target

    def get_optimum_path(self) -> Path:
        self._ensure_initialized()
        assert self.filename_constant is not None
        optimum = resolve_optimum_file(self.filename_constant)
        optimum.parent.mkdir(parents=True, exist_ok=True)
        return optimum

    def cleanup_temp_dir(self) -> None:
        cleanup_temp(self)

    def commit_children(self, expected_size: int, retries: int = 3) -> None:
        self._ensure_initialized()
        assert self.filename_constant is not None
        commit_children(
            layout=self,
            file_name=self.filename_constant,
            expected_size=expected_size,
            retries=retries,
        )


# Ensure legacy imports resolve to the compatibility class.
compat_module = ModuleType("src.classes.PathResolver")
compat_module.PathResolver = CompatPathResolver
sys.modules.setdefault("src.classes.PathResolver", compat_module)


@pytest.fixture
def test_only_pathresolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> CompatPathResolver:
    """Create a PathResolver rooted at the pytest temp directory."""
    monkeypatch.setattr(CompatPathResolver, "PROJECT_ROOT", tmp_path, raising=False)
    path_resolver = CompatPathResolver(tmp_path)
    path_resolver.initialize(filename_constant="pytest_temp_file")
    assert tmp_path == path_resolver.PROJECT_ROOT
    return path_resolver


@pytest.fixture
def test_only_rng() -> np.random.Generator:
    """Return a deterministic RNG for tests."""
    rng = np.random.default_rng(seed=1234)
    return rng


@pytest.fixture
def root_path() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def temp_file(test_only_pathresolver: CompatPathResolver) -> Path:
    """Return a temporary output file path from the mocked resolver."""
    temp_dir = test_only_pathresolver.get_temp_path()
    temp_filename = test_only_pathresolver.filename_constant
    assert temp_filename is not None
    return temp_dir / temp_filename


@pytest.fixture
def storage_root(tmp_path: Path) -> Path:
    """Return the root directory used by storage-layer tests."""
    return tmp_path / "storage_root"


@pytest.fixture
def storage_layout(storage_root: Path) -> CorePathResolver:
    """Provide a prepared layout under a temporary experiment root."""
    layout = CorePathResolver(storage_root)
    ensure_layout_paths(layout)
    return layout


@pytest.fixture
def experiment_storage(storage_root: Path) -> ExperimentStorage:
    """Return an ExperimentStorage instance rooted in a temporary directory."""
    return ExperimentStorage.from_root(
        root=storage_root, filename="population", data_file_name="f_dataset"
    )


@pytest.fixture
def base_input_config_data() -> dict:
    """Return a valid input config payload for GA runs."""
    return {
        "data": {"data_filename": "items.csv", "max_weight": 100},
        "population": {"size": 10, "generations": 5, "stream_batch_size": 2},
        "selection": {
            "type": "rank",
            "selection_pressure": 1.5,
            "tournament_size": None,
        },
        "genetic_operators": {
            "crossover_type": "one",
            "crossover_probability": 0.6,
            "mutation_probability": 0.1,
            "penalty_multiplier": 2.0,
            "strict_weight_constraints": False,
        },
        "experiment": {"seed": 123, "identifier": "exp-1", "log_level": "INFO"},
    }


@pytest.fixture
def input_config(base_input_config_data: dict) -> InputConfig:
    """Provide a validated InputConfig instance."""
    return InputConfig(**base_input_config_data)


@pytest.fixture
def experiment_config(
    input_config: InputConfig, tmp_path: Path
) -> CoreExperimentConfig:
    """Provide a runtime ExperimentConfig instance with temp root."""
    return CoreExperimentConfig(
        input=input_config,
        job_id="job-123",
        root_path=tmp_path / "experiment",
    )


@pytest.fixture
def clean_experiment_logger() -> logging.Logger:
    """Remove handlers from the experiment logger before and after a test."""
    logger = logging.getLogger("ga_core.experiment")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)
    yield logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    logger.setLevel(logging.NOTSET)
