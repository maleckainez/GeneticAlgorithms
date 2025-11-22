"""Configure logging and provide helper functions for experiment runtime logs."""

import logging
from logging import Logger, LoggerAdapter
from pathlib import Path

from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver


def initialize(config: ExperimentConfig, paths: PathResolver) -> LoggerAdapter:
    """Initialize experiment logger and wrap it in a LoggerAdapter.

    Sets up console and file handlers, applies a unified formatter and attaches
    experiment identifier as contextual information.

    Args:
        config (ExperimentConfig): Experiment configuration with logging settings
            (level, experiment identifier).
        paths (PathResolver): Resolver providing path for log file output.

    Returns:
        logging.LoggerAdapter: Configured logger adapter with exp_id in context.
    """
    log_level = config.log_level
    log_path = paths.get_logging_path()
    exp_identifier = config.experiment_identifier
    filepath = Path(log_path / f"runtime_experiment_{exp_identifier}.log")

    try:
        level = logging.getLevelName(log_level.upper())
    except ValueError:
        level = logging.INFO

    main_logger = logging.getLogger("GA experiment run")
    main_logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s][experiment number %(exp_id)s] %(message)s"
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

    adapter = logging.LoggerAdapter(main_logger, {"exp_id": exp_identifier})
    adapter.debug("--- LOG CONFIG FINISHED ---")
    adapter.debug(f"Log files saved as: runtime_experiment_{exp_identifier}.log\n")
    return adapter


def log_generation(
    logger: Logger,
    best_idx: int,
    best_score: int,
    weight: int,
    iteration: int,
    repetitions: int,
) -> None:
    """Log summary information for a single generation.

    Args:
        logger (Logger): Logger or logger adapter used for writing the message.
        best_idx (int): Index of the best individual in the population.
        best_score (int): Fitness score of the best individual.
        weight (int): Weight of the best individual.
        iteration (int): Current generation number.
        repetitions (int): Count of individuals sharing the best score.
    """
    if repetitions > 0:
        repetition_msg = f"      There were {repetitions} individuals with same scores"
    else:
        repetition_msg = " "
    logger.info(
        f"Generation {iteration}: \n"
        f"      Index of best individual: {best_idx}\n"
        f"      Fitness of best individual: {best_score}\n"
        f"      Weight of best individual: {weight}\n"
        f"{repetition_msg}"
    )


generation = log_generation
