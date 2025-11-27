"""Defines I/O helper functions for loading problem data and configuration."""

import os
from pathlib import Path

import numpy as np
import yaml

from src.config.schemas import JobConfig


def load_data(path: str | Path) -> np.ndarray:
    """Load item data from a text file.

    Each line in the file must contain two numeric values: ``<value> <weight>``.
    Every line represents one item.

    Args:
        path (str | pathlib.Path): Path to the text file containing item data.

    Returns:
        np.ndarray: A 2D array of shape ``(items, 2)`` and dtype ``np.int64``.
        Each line corresponds to individual item where column 0 contains values
        and column 1 weights of those items.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file is empty, contains non-numeric data, has missing
            fields, or is incorrectly formatted.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found {path}")
    with open(path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if not lines:
        raise ValueError("File is empty")
    data_in_lines = []
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"Invalid values on line {i+1}: expected 2 values, got {len(parts)}"
            )
        try:
            value, weight = map(int, parts)
        except ValueError:
            raise ValueError(
                f"Invalid values on line {i+1}: received non integer input {line}"
            )
        data_in_lines.append([value, weight])
        items = np.array(data_in_lines, dtype=np.int64)
    return items


def load_yaml_config(filepath: Path | str) -> dict:
    """Load experiment configuration from a YAML file.

    The YAML file is expected to follow the internal configuration schema, with
    top-level sections such as ``data``, ``population``, ``selection``,
    ``genetic_operators``, and ``experiment``. Selected fields are extracted
    and normalized into a flat dictionary compatible with ``ExperimentConfig``.

    Args:
        filepath (pathlib.Path | str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing normalized configuration fields, including
        entries such as ``data_filename``, ``max_weight``, ``population_size``,
        ``generations``, ``stream_batch_size``, ``selection_type``,
        ``selection_pressure``, ``crossover_type``, ``crossover_probability``,
        ``mutation_probability``, ``penalty``, ``seed``,
        ``experiment_identifier``, and ``log_level``.
    """
    with open(filepath, "r") as file:
        yaml_file = yaml.safe_load(file)

    job = JobConfig.model_validate(yaml_file)

    return {
        "data_filename": job.data.filename,
        "max_weight": job.data.max_weight,
        "population_size": job.population.size,
        "generations": job.population.generations,
        "stream_batch_size": job.population.stream_batch_size,
        "selection_type": job.selection.type.value,  # Enum → str
        "selection_pressure": job.selection.selection_pressure,
        "crossover_type": job.genetic_operators.crossover_type.value,
        "crossover_probability": job.genetic_operators.crossover_probability,
        "mutation_probability": job.genetic_operators.mutation_probability,
        "penalty": job.genetic_operators.penalty_multiplier,
        "seed": job.experiment.seed,
        "experiment_identifier": job.experiment.identifier,
        "log_level": job.experiment.log_level.value,  # Enum → str
    }
