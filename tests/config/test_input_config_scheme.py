"""Tests for input configuration validation models."""

import copy

import pytest
from pydantic import ValidationError
from src.ga_core.config import InputConfig
from src.ga_core.config.input_config_scheme import (
    GeneticOperatorsConfig,
    SelectionType,
)


def test_input_config_accepts_valid_payload(base_input_config_data: dict) -> None:
    config = InputConfig(**base_input_config_data)

    assert config.population.size == 10
    assert config.selection.type == SelectionType.LINEAR_RANK
    assert config.genetic_operators.penalty_multiplier == 2.0


def test_population_size_must_be_even(base_input_config_data: dict) -> None:
    payload = copy.deepcopy(base_input_config_data)
    payload["population"]["size"] = 11

    with pytest.raises(ValidationError):
        InputConfig(**payload)


def test_rank_selection_requires_pressure(base_input_config_data: dict) -> None:
    payload = copy.deepcopy(base_input_config_data)
    payload["selection"]["selection_pressure"] = None

    with pytest.raises(ValidationError, match="Selection pressure"):
        InputConfig(**payload)


def test_rank_selection_pressure_bounds(base_input_config_data: dict) -> None:
    payload = copy.deepcopy(base_input_config_data)
    payload["selection"]["selection_pressure"] = 2.5

    with pytest.raises(ValidationError):
        InputConfig(**payload)


def test_tournament_selection_requires_size(base_input_config_data: dict) -> None:
    payload = copy.deepcopy(base_input_config_data)
    payload["selection"]["type"] = SelectionType.TOURNAMENT.value
    payload["selection"]["selection_pressure"] = None
    payload["selection"]["tournament_size"] = None

    with pytest.raises(ValidationError, match="Tournament size"):
        InputConfig(**payload)


def test_penalty_set_to_zero_when_strict_constraints_enabled() -> None:
    config = GeneticOperatorsConfig(
        crossover_type="one",
        crossover_probability=0.7,
        mutation_probability=0.2,
        penalty_multiplier=5.0,
        strict_weight_constraints=True,
    )

    assert config.penalty_multiplier == 0.0
