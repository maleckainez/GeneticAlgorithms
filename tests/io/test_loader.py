"""Tests for I/O loader helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.ga_core.io.loader import (
    load_experiment_data,
    load_yaml_config,
    read_optimum,
)


def test_load_experiment_data_reads_values(tmp_path: Path) -> None:
    data_file = tmp_path / "items.txt"
    data_file.write_text("5 3\n7 4\n")

    data = load_experiment_data(data_file)

    np.testing.assert_array_equal(data, np.array([[5, 3], [7, 4]], dtype=np.int64))


def test_load_experiment_data_raises_on_empty_file(tmp_path: Path) -> None:
    data_file = tmp_path / "items.txt"
    data_file.write_text("")

    with pytest.raises(ValueError, match="empty"):
        load_experiment_data(data_file)


def test_load_yaml_config_validates_schema(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "data:",
                "  data_filename: items.csv",
                "  max_weight: 50",
                "population:",
                "  size: 10",
                "  generations: 5",
                "  stream_batch_size: 2",
                "selection:",
                "  type: rank",
                "  selection_pressure: 1.5",
                "  tournament_size: null",
                "genetic_operators:",
                "  crossover_type: one",
                "  crossover_probability: 0.6",
                "  mutation_probability: 0.1",
                "  penalty_multiplier: 2.0",
                "  strict_weight_constraints: false",
                "experiment:",
                "  seed: 123",
                "  identifier: exp-1",
                "  log_level: INFO",
            ]
        )
    )

    config = load_yaml_config(yaml_path)

    assert config.data.data_filename == "items.csv"
    assert config.population.size == 10
    assert config.genetic_operators.crossover_type.value == "one"


def test_read_optimum_returns_first_value(tmp_path: Path) -> None:
    opt_path = tmp_path / "optimum.csv"
    pd.DataFrame([[42]]).to_csv(opt_path, header=False, index=False)

    optimum = read_optimum(opt_path)

    assert optimum == 42
