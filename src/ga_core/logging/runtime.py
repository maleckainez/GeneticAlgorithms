"""Logger initialization helpers for experiment runs."""

import logging
from logging import LoggerAdapter
from pathlib import Path

from src.ga_core.config import ExperimentConfig
from src.ga_core.storage import StorageLayout

_LOGGER_NAME = "ga_core.experiment"


def _resolve_level(log_level: str) -> int:
    """Return numeric logging level or raise on invalid input.

    Args:
        log_level: Textual logging level, case-insensitive.

    Raises:
        ValueError: If the provided level cannot be resolved to a number.

    Returns:
        int: Numeric level understood by `logging`.
    """
    level = logging.getLevelName(log_level.upper())
    if isinstance(level, str):
        raise ValueError("Logger level was set incorrectly")
    return level


def initialize(log_level: str, log_path: Path, file_name: str) -> LoggerAdapter:
    """Initialize experiment logger and wrap it in a LoggerAdapter.

    Sets up console and file handlers and applies a unified formatter.

    Args:
        log_level: Textual logging level (e.g. "INFO", "DEBUG").
        log_path: Directory where the log file should be created.
        file_name: Base name used to construct the log file name.

    Returns:
        LoggerAdapter: Configured adapter instance for this experiment.
    """
    filepath = log_path / f"runtime_experiment_{file_name}.log"

    level = _resolve_level(log_level=log_level)

    main_logger = logging.getLogger(_LOGGER_NAME)
    main_logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s][%(experiment_name)s] %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    if not main_logger.handlers:
        main_logger.addHandler(console_handler)
        main_logger.addHandler(file_handler)

    adapter = logging.LoggerAdapter(main_logger, {"experiment_name": file_name})
    adapter.debug("--- LOG CONFIG FINISHED ---")
    adapter.debug(f"Log files saved as: runtime_experiment_{file_name}.log\n")
    return adapter


def from_config_and_layout(
    config: ExperimentConfig, layout: StorageLayout
) -> LoggerAdapter:
    """Initialize a logger using experiment config and storage layout."""
    return initialize(
        log_level=config.input.experiment.log_level,
        log_path=layout.logs,
        file_name=config.job_id,
    )


def from_config(config: ExperimentConfig, log_path: Path) -> LoggerAdapter:
    """Initialize a logger using experiment config and an explicit log path."""
    return initialize(
        log_level=config.input.experiment.log_level,
        log_path=log_path,
        file_name=config.job_id,
    )


def from_layout(log_level: str, layout: StorageLayout, file_name: str):
    """Initialize a logger using only a layout and explicit parameters."""
    return initialize(
        log_level=log_level,
        log_path=layout.logs,
        file_name=file_name,
    )
