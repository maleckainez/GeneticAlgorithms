"""I/O helpers for loading problem data and experiment configuration."""

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pandas import read_csv

from src.ga_core.config import InputConfig
from src.ga_core.logging import LoggerType


def load_experiment_data(data_path: Path, logger: LoggerType = None) -> np.ndarray:
    """Load item data from a two-column text file.

    Each non-empty line must contain ``<value> <weight>``. The output array has
    shape ``(items, 2)`` with values in column 0 and weights in column 1.

    Args:
        data_path: Path to the text file containing item data.
        logger: Optional logger for reporting errors or progress.

    Returns:
        np.ndarray: Array of item value/weight pairs with dtype ``int64``.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file is empty, contains non-numeric data, has missing
            fields, or is incorrectly formatted.
    """
    if not data_path.exists():
        if logger is not None:
            logger.error("Error while loading file containing data: %s", data_path)
        raise FileNotFoundError(f"File not found {data_path}")
    with open(data_path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if not lines:
        if logger is not None:
            logger.error("Error while loading data file. File %s is empty.", data_path)
        raise ValueError("File is empty")
    data_in_lines = []
    if logger is not None:
        logger.debug("Data file %s opened. Reading data started.", data_path)
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 2:
            if logger is not None:
                logger.error(
                    "Error while reading data file %s. "
                    "Loader found invalid values on line %d:"
                    " expecting 2 values in line, found %d",
                    data_path,
                    i + 1,
                    len(parts),
                )
            raise ValueError(
                f"Invalid values on line {i+1}: expected 2 values, got {len(parts)}"
            )
        try:
            value, weight = map(int, parts)
        except ValueError:
            if logger is not None:
                logger.error(
                    "Error while reading data file %s. "
                    "Loader found invalid values on line %d,"
                    " received non integer input",
                    data_path,
                    i + 1,
                    line,
                )
            raise ValueError(
                f"Invalid values on line {i+1}: received non integer input {line}"
            )
        data_in_lines.append([value, weight])
    items = np.array(data_in_lines, dtype=np.int64)
    if logger is not None:
        logger.debug("Data file %s loaded successfully.", data_path)
    return items


def load_yaml_config(yaml_path: Path, logger: LoggerType = None) -> InputConfig:
    """Load experiment configuration from a YAML file.

    The YAML file is expected to follow the internal configuration schema, with
    top-level sections such as ``data``, ``population``, ``selection``,
    ``genetic_operators``, and ``experiment``. Selected fields are extracted
    and normalized into a flat dictionary compatible with ``ExperimentConfig``.

    Args:
        yaml_path: Path to the YAML configuration file.
        logger: Optional logger for reporting errors or progress.

    Returns:
        InputConfig: Validated configuration object.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValidationError: If the YAML content does not satisfy the schema.
    """
    if not yaml_path.exists():
        if logger is not None:
            logger.error("Configuration was not found. File %s not found", yaml_path)
        raise FileNotFoundError(f"Configuration file {yaml_path} not found.")
    with open(yaml_path, "r") as file:
        yaml_file = yaml.safe_load(file)
    if logger is not None:
        logger.debug("Config file opened successfully")
    loaded_input_config = InputConfig.model_validate(yaml_file)
    if logger is not None:
        logger.debug("Config file validated")
    return loaded_input_config


def read_optimum(optimum_path: Path, logger: LoggerType = None) -> Any:
    """Load optimum value from a CSV-like file.

    The file is expected to contain the optimum in the first cell.

    Args:
        optimum_path: Path to the file with optimum value.
        logger: Optional logger for reporting errors or progress.

    Returns:
        float | int: Parsed optimum value.

    Raises:
        FileNotFoundError: If the optimum file does not exist.
    """
    if not optimum_path.exists():
        if logger is not None:
            logger.error("File containing optimum value: %s not found.", optimum_path)
        raise FileNotFoundError(
            f"File containing optimum value: {optimum_path} not found."
        )
    file = read_csv(optimum_path, header=None)
    if logger is not None:
        logger.debug("Optimum containing file read successfully.")
    optimum_val = file.iloc[0, 0]
    if logger is not None:
        logger.debug("Optimum value extracted.")
    return optimum_val
