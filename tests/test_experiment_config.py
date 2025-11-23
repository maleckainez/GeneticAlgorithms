"""Validates ExperimentConfig initialization constraints and defaults."""

import numpy as np
import pytest
from src.classes.ExperimentConfig import ExperimentConfig


def _base_kwargs() -> dict:
    return {
        "data_filename": "items.csv",
        "population_size": 10,
        "generations": 2,
        "max_weight": 50,
        "seed": 1,
        "selection_type": "roulette",
        "crossover_type": "one",
        "crossover_probability": 0.5,
        "mutation_probability": 0.1,
        "penalty": 2.0,
        "experiment_identifier": 1,
        "log_level": "INFO",
        "stream_batch_size": 5,
    }


def test_population_size_must_be_even() -> None:
    kwargs = _base_kwargs()
    kwargs["population_size"] = 3
    with pytest.raises(ValueError, match="Population has to be even"):
        ExperimentConfig(**kwargs)


def test_crossover_probability_bounds() -> None:
    kwargs = _base_kwargs()
    kwargs["crossover_probability"] = 1.5
    with pytest.raises(
        ValueError, match="Crossover_probability must be between 0 and 1"
    ):
        ExperimentConfig(**kwargs)


def test_mutation_probability_bounds() -> None:
    kwargs = _base_kwargs()
    kwargs["mutation_probability"] = -0.01
    with pytest.raises(
        ValueError, match="Mutation probability must be between 0 and 1"
    ):
        ExperimentConfig(**kwargs)


def test_max_weight_must_be_positive() -> None:
    kwargs = _base_kwargs()
    kwargs["max_weight"] = -1
    with pytest.raises(ValueError, match="Max weight must be positive"):
        ExperimentConfig(**kwargs)


def test_population_size_must_be_positive() -> None:
    kwargs = _base_kwargs()
    kwargs["population_size"] = 0
    with pytest.raises(ValueError, match="Population size must be greater than 0"):
        ExperimentConfig(**kwargs)


def test_generations_must_be_positive() -> None:
    kwargs = _base_kwargs()
    kwargs["generations"] = 0
    with pytest.raises(ValueError, match="Generations must be greater than 1"):
        ExperimentConfig(**kwargs)


def test_selection_pressure_range_for_rank_selection() -> None:
    kwargs = _base_kwargs()
    kwargs["selection_type"] = "rank"
    kwargs["selection_pressure"] = 2.5
    with pytest.raises(
        ValueError, match="Selection pressure must be float in range from 1 to 2"
    ):
        ExperimentConfig(**kwargs)


def test_none_rng_and_seed_none() -> None:
    kwargs = _base_kwargs()
    kwargs["seed"] = None
    assert (ExperimentConfig(**kwargs)).rng is not None


def test_existing_rng_and_none_pressure() -> None:
    kwargs = _base_kwargs()
    kwargs["selection_type"] = "rank"
    kwargs["selection_pressure"] = None
    rng = np.random.default_rng()
    config = ExperimentConfig(rng=rng, **kwargs)
    assert config.rng is not None
    assert config.selection_pressure == 1


def test_porobability_of_failure_q_over_max() -> float:
    kwargs = _base_kwargs()
    config = ExperimentConfig(**kwargs)
    weight_sum = 1
    q = config.generate_probability_of_failure(weight_sum)
    assert q is not None
    # 50/1 = 50, 50 is more than 1 (max), assert that q equals max
    assert q == 1


def test_porobability_of_failure_q_below_max() -> float:
    kwargs = _base_kwargs()
    config = ExperimentConfig(**kwargs)
    weight_sum = 5000
    q = config.generate_probability_of_failure(weight_sum)
    assert q is not None
    assert q == 0.01


def test_porobability_of_failure_wrong_weight_sum() -> float:
    kwargs = _base_kwargs()
    config = ExperimentConfig(**kwargs)
    weight_sum = 0.5
    with pytest.raises(ValueError, match="Weight sum must be greater than 0"):
        config.generate_probability_of_failure(weight_sum)
