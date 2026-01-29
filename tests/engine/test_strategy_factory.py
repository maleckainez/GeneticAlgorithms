"""Unit tests for strategy executor factory."""

from pathlib import Path

import pytest
from src.ga_core.config.experiment_config import ExperimentConfig
from src.ga_core.config.input_config_scheme import (
    DataConfig,
    EncodingType,
    ExperimentVals,
    GeneticOperatorsConfig,
    InputConfig,
    PopulationConfig,
    SelectionConfig,
)
from src.ga_core.engine.strategies.factory import create_strategy_executor


@pytest.fixture
def sample_config():
    """Create sample experiment config."""
    input_config = InputConfig(
        data=DataConfig(data_filename="test.txt", max_weight=100),
        population=PopulationConfig(size=10, generations=5),
        selection=SelectionConfig(
            type="roulette", selection_pressure=None, tournament_size=None
        ),
        genetic_operators=GeneticOperatorsConfig(
            crossover_type="one",
            crossover_probability=0.8,
            mutation_probability=0.01,
            penalty_multiplier=1.5,
        ),
        experiment=ExperimentVals(seed=42, log_level="INFO"),
    )

    config = ExperimentConfig(
        input=input_config, job_id="test_job", root_path=Path("/tmp")
    )

    return config


def test_create_strategy_executor(sample_config):
    """Factory creates strategy executor from config."""
    executor = create_strategy_executor(sample_config)

    assert executor is not None
    assert executor.population_size == 10
    assert executor.selection_fn is not None
    assert executor.pairing_fn is not None
    assert executor.reproduction_executor is not None
    assert executor.fitness_fn is not None


def test_create_strategy_executor_with_tournament(sample_config):
    """Factory creates executor with tournament selection."""
    sample_config.input.selection.type = "tournament"
    sample_config.input.selection.tournament_size = 5

    executor = create_strategy_executor(sample_config)

    assert executor is not None
    assert executor.selection_fn is not None


def test_create_strategy_executor_with_rank_selection(sample_config):
    """Factory creates executor with rank selection."""
    sample_config.input.selection.type = "rank"
    sample_config.input.selection.selection_pressure = 1.8

    executor = create_strategy_executor(sample_config)

    assert executor is not None
    assert executor.selection_fn is not None


def test_create_strategy_executor_with_two_point_crossover(sample_config):
    """Factory creates executor with two-point crossover."""
    sample_config.input.genetic_operators.crossover_type = "two"

    executor = create_strategy_executor(sample_config)

    assert executor is not None
    assert executor.reproduction_executor is not None


def test_create_strategy_executor_with_sequential_pairing(sample_config):
    """Factory creates executor with sequential pairing."""
    executor = create_strategy_executor(sample_config, pairing_strategy="sequential")

    assert executor is not None
    assert executor.pairing_fn is not None


def test_create_strategy_executor_respects_encoding(sample_config):
    """Factory respects encoding parameter."""
    executor = create_strategy_executor(sample_config, encoding=EncodingType.BINARY)

    assert executor is not None
    assert executor.fitness_fn is not None
