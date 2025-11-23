"""Tests for in-memory child population memmap creation and cleanup."""

import numpy as np
from src.classes.ChildrenHandler import ChildrenHandler


def test_children_handler_creates_memmap_and_closes(
    experiment_config_factory, test_only_pathresolver
) -> None:
    config = experiment_config_factory(
        population_size=4,
        generations=2,
        max_weight=10,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.5,
        mutation_probability=0.1,
        penalty_multiplier=1.0,
        stream_batch=2,
    )
    handler = ChildrenHandler(
        config=config,
        paths=test_only_pathresolver,
        genome_length=3,
    )
    children = handler.get_children_handle()
    assert children.shape == (4, 3)
    assert children.dtype == np.uint8

    child_path = test_only_pathresolver.get_children_filepath()
    assert child_path.exists()
    expected_size = config.population_size * 3
    assert child_path.stat().st_size == expected_size

    handler.close()
    assert handler.children_handle is None
