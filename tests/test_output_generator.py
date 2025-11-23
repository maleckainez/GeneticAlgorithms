"""Tests for CSV output writer used during experiment logging."""

import csv

import pytest
from src.classes.OutputGenerator import OutputGenerator


def test_output_generator_raises_when_not_initialized(
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
    )
    generator = OutputGenerator(test_only_pathresolver, config)
    with pytest.raises(RuntimeError, match="Plotter not opened"):
        generator.write_iteration(
            iteration=0,
            best_fitness=1,
            best_weight=2,
            avg_fitness=1.5,
            worst_fitness=0,
            worst_weight=1,
            identical_best_count=0,
            genome="1010",
        )


def test_output_generator_writes_meta_and_rows(
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
    )
    generator = OutputGenerator(test_only_pathresolver, config)
    generator.init_csv(config)
    generator.write_iteration(
        iteration=1,
        best_fitness=10,
        best_weight=5,
        avg_fitness=6.5,
        worst_fitness=2,
        worst_weight=3,
        identical_best_count=0,
        genome="1010",
    )
    generator.close()
    with open(generator.filename, newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["# data_filename", config.data_filename]
    assert rows[-1] == [
        "1",
        "10",
        "5",
        "6.5",
        "2",
        "3",
        "0",
        "1010",
    ]


def test_output_generator_opens_and_closes_correctly(
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
    )
    generator = OutputGenerator(test_only_pathresolver, config)
    generator._open()
    generator._open()
    generator.init_csv(config)
    generator.write_iteration(
        iteration=1,
        best_fitness=10,
        best_weight=5,
        avg_fitness=6.5,
        worst_fitness=2,
        worst_weight=3,
        identical_best_count=0,
        genome="1010",
    )
    generator.close()
    generator.close()
