"""Tests for PopulationManager orchestration of population and children buffers."""

from pathlib import Path

import numpy as np
import pytest
from src.ga_core.config.input_config_scheme import SwapType
from src.ga_core.engine.population.directory_manager import DirectoryManager
from src.ga_core.engine.population.manager import PopulationManager
from src.ga_core.storage import ExperimentStorage
from src.ga_core.storage.directory_utils import ensure_layout_paths


def test_manager_initialization_sets_configuration() -> None:
    """Test that PopulationManager initialization stores configuration."""
    pop_mgr = PopulationManager(
        population_size=10,
        genome_length=5,
        stream_batch_size=2,
        storage=None,  # type: ignore
        rng=np.random.default_rng(42),
        overweight_probability=0.5,
        commit_mode=SwapType.FLIP,
    )

    assert pop_mgr._pop_size == 10
    assert pop_mgr._gen_len == 5
    assert pop_mgr._batch == 2
    assert pop_mgr._q == 0.5
    assert pop_mgr._swap_mode == SwapType.FLIP


def test_population_property_raises_before_init() -> None:
    """Test that population property raises RuntimeError if not initialized."""
    pop_mgr = PopulationManager(
        population_size=10,
        genome_length=5,
        stream_batch_size=2,
        storage=None,  # type: ignore
        rng=np.random.default_rng(42),
        overweight_probability=0.5,
    )

    with pytest.raises(RuntimeError, match="Population not initialized"):
        _ = pop_mgr.population


def test_children_property_raises_before_init() -> None:
    """Test that children property raises RuntimeError if not initialized."""
    pop_mgr = PopulationManager(
        population_size=10,
        genome_length=5,
        stream_batch_size=2,
        storage=None,  # type: ignore
        rng=np.random.default_rng(42),
        overweight_probability=0.5,
    )

    with pytest.raises(RuntimeError, match="Children not initialized"):
        _ = pop_mgr.children


def _create_storage(tmp_path: Path, job_id: str) -> ExperimentStorage:
    """Helper to create ExperimentStorage for tests."""
    layout = DirectoryManager(root=tmp_path)
    ensure_layout_paths(layout)
    return ExperimentStorage(
        layout=layout,
        job_id=job_id,
        data_file_name="knapsack_small",
    )


def test_initialize_population_creates_array(tmp_path: Path) -> None:
    """Test that initialize_population creates the population buffer."""
    storage = _create_storage(tmp_path, "test_pop")
    rng = np.random.default_rng(42)

    pop_mgr = PopulationManager(
        population_size=5,
        genome_length=10,
        stream_batch_size=2,
        storage=storage,
        rng=rng,
        overweight_probability=0.3,
    )

    pop_mgr.initialize_population()
    population = pop_mgr.population

    assert population is not None
    assert population.shape == (5, 10)
    assert population.dtype == np.uint8


def test_initialize_children_creates_array(tmp_path: Path) -> None:
    """Test that initialize_children creates the children buffer."""
    storage = _create_storage(tmp_path, "test_child")

    pop_mgr = PopulationManager(
        population_size=5,
        genome_length=10,
        stream_batch_size=2,
        storage=storage,
        rng=np.random.default_rng(42),
        overweight_probability=0.3,
    )

    pop_mgr.initialize_children()
    children = pop_mgr.children

    assert children is not None
    assert children.shape == (5, 10)
    assert children.dtype == np.uint8


def test_init_pop_and_children_initializes_both(tmp_path: Path) -> None:
    """Test that init_pop_and_children initializes both buffers."""
    storage = _create_storage(tmp_path, "test_both")
    rng = np.random.default_rng(42)

    pop_mgr = PopulationManager(
        population_size=5,
        genome_length=10,
        stream_batch_size=2,
        storage=storage,
        rng=rng,
        overweight_probability=0.3,
    )

    pop_mgr.init_pop_and_children()

    assert pop_mgr.population is not None
    assert pop_mgr.children is not None
    assert pop_mgr.population.shape == (5, 10)
    assert pop_mgr.children.shape == (5, 10)


def test_initialize_population_twice_raises_error(tmp_path: Path) -> None:
    """Test that calling initialize_population twice raises RuntimeError."""
    storage = _create_storage(tmp_path, "test_double_pop")

    pop_mgr = PopulationManager(
        population_size=5,
        genome_length=10,
        stream_batch_size=2,
        storage=storage,
        rng=np.random.default_rng(42),
        overweight_probability=0.3,
    )

    pop_mgr.initialize_population()
    with pytest.raises(RuntimeError, match="Population already initialized"):
        pop_mgr.initialize_population()


def test_initialize_children_twice_raises_error(tmp_path: Path) -> None:
    """Test that calling initialize_children twice raises RuntimeError."""
    storage = _create_storage(tmp_path, "test_double_child")

    pop_mgr = PopulationManager(
        population_size=5,
        genome_length=10,
        stream_batch_size=2,
        storage=storage,
        rng=np.random.default_rng(42),
        overweight_probability=0.3,
    )

    pop_mgr.initialize_children()
    with pytest.raises(RuntimeError, match="Children already initialized"):
        pop_mgr.initialize_children()


def test_commit_with_flip_mode_swaps_buffers(tmp_path: Path) -> None:
    """Test that FLIP mode swaps population and children in memory."""
    storage = _create_storage(tmp_path, "test_flip")
    rng = np.random.default_rng(42)

    pop_mgr = PopulationManager(
        population_size=5,
        genome_length=10,
        stream_batch_size=2,
        storage=storage,
        rng=rng,
        overweight_probability=0.3,
        commit_mode=SwapType.FLIP,
    )

    pop_mgr.init_pop_and_children()

    # Modify population to track it
    pop_mgr.population[0, 0] = 1
    pop_before_commit = pop_mgr.population.copy()

    pop_mgr.commit_reproduction_of_population()

    # After flip, children should have the old population values
    assert np.array_equal(pop_mgr.children, pop_before_commit)


def test_commit_raises_if_not_initialized(tmp_path: Path) -> None:
    """Test that commit raises RuntimeError if buffers not initialized."""
    storage = _create_storage(tmp_path, "test_not_init")

    pop_mgr = PopulationManager(
        population_size=5,
        genome_length=10,
        stream_batch_size=2,
        storage=storage,
        rng=np.random.default_rng(42),
        overweight_probability=0.3,
    )

    with pytest.raises(RuntimeError, match="Population or children not initialized"):
        pop_mgr.commit_reproduction_of_population()
