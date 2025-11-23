"""Tests for human-readable experiment name generation utility."""

import datetime
from collections.abc import Callable

import pytest
from src.classes.ExperimentConfig import ExperimentConfig
from src.methods.name_generator import name_generator


def _freeze_time(monkeypatch: pytest.MonkeyPatch, frozen: datetime.datetime) -> None:

    class FrozenDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz: datetime.tzinfo | None = None) -> datetime.datetime:
            if tz is None:
                return frozen
            return frozen.astimezone(tz)

    monkeypatch.setattr("src.methods.name_generator.datetime.datetime", FrozenDateTime)


def test_name_generator_builds_expected_identifier(
    monkeypatch: pytest.MonkeyPatch,
    experiment_config_factory: Callable[..., ExperimentConfig],
) -> None:

    frozen_time = datetime.datetime(2024, 1, 2, 3, 4, 5)
    _freeze_time(monkeypatch, frozen_time)

    config = experiment_config_factory(
        population_size=10,
        generations=5,
        max_weight=100,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.8,
        mutation_probability=0.05,
        penalty_multiplier=2.0,
        exp_identifier=0,
    )

    genome_length = 42
    generated = name_generator(config, genome_length)

    expected = "temp_test_data_csv-roulette-PS10-GW42-GE5-CR0p8-" "MR0p05-EXP000T0405"
    assert generated == expected


def test_name_generator_sanitizes_filename_and_formats_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    frozen_time = datetime.datetime(2024, 6, 7, 8, 10, 11)
    _freeze_time(monkeypatch, frozen_time)

    config = ExperimentConfig(
        data_filename="data file-1.csv",
        population_size=4,
        generations=3,
        max_weight=50,
        seed=1,
        selection_type="rank",
        crossover_type="two",
        crossover_probability=0.123,
        mutation_probability=0.9,
        penalty=2.0,
        experiment_identifier=7,
        log_level="INFO",
        stream_batch_size=5,
        selection_pressure=1.5,
    )

    genome_length = 8
    generated = name_generator(config, genome_length)

    assert generated.startswith("data_file_1_csv-rank-PS4-GW8-GE3-CR0p123-MR0p9-")
    assert generated.endswith("EXP007T1011")
