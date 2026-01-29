"""Tests for runtime ExperimentConfig helpers."""

import copy
from pathlib import Path

import numpy as np
import pytest
from numpy.random import Generator
from src.ga_core.config import ExperimentConfig, InputConfig


def test_create_rng_is_deterministic_with_seed(
    experiment_config: ExperimentConfig,
) -> None:
    rng_one = experiment_config.create_rng()
    rng_two = experiment_config.create_rng()

    assert isinstance(rng_one, Generator)
    assert rng_one.integers(0, 10_000) == rng_two.integers(0, 10_000)


def test_create_rng_without_seed_uses_random_seed(
    base_input_config_data: dict, tmp_path: Path
) -> None:
    payload = copy.deepcopy(base_input_config_data)
    payload["experiment"]["seed"] = None
    config = InputConfig(**payload)
    runtime_config = ExperimentConfig(
        input=config, job_id="job-999", root_path=tmp_path
    )

    first = runtime_config.create_rng().integers(0, 10_000, size=3)
    second = runtime_config.create_rng().integers(0, 10_000, size=3)
    assert not np.array_equal(first, second)


def test_estimate_overweight_probability_clamps_to_range(
    experiment_config: ExperimentConfig,
) -> None:
    high = experiment_config.estimate_overweight_probability(total_item_weight=50)
    mid = experiment_config.estimate_overweight_probability(total_item_weight=200)

    assert high == 1.0
    assert mid == pytest.approx(0.5)


def test_estimate_overweight_probability_raises_on_non_positive(
    experiment_config: ExperimentConfig,
) -> None:
    with pytest.raises(ValueError):
        experiment_config.estimate_overweight_probability(0)


def test_as_dict_contains_input_and_job_id(
    experiment_config: ExperimentConfig,
) -> None:
    data = experiment_config.as_dict()

    assert data["job_id"] == "job-123"
    assert data["input"]["data"]["data_filename"] == "items.csv"
