import logging
from logging import Logger
from pathlib import Path
from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver


def initialize(config: ExperimentConfig, paths: PathResolver):
    log_level = config.log_level
    log_path = paths.get_logging_path()
    exp_identifier = config.experiment_identifier
    filepath = Path(log_path / f"runtime_experiment_{exp_identifier}.log")

    try:
        level = logging.getLevelName(log_level.upper())
    except ValueError:
        level = logging.INFO

    main_logger = logging.getLogger(str(exp_identifier))
    main_logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s:\n%(message)s"
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

    main_logger.debug("--- LOG CONFIG FINISHED ---")
    main_logger.debug(f"Log files saved as: runtime_experiment_{exp_identifier}.log\n")
    return main_logger


def generation(logger: Logger, best_idx: int, best_score: int, weight: int):
    logger.info(
        f"Population created successfully as iteration 0\n"
        f"Generation 0: \n"
        f"      Index of best individual: {best_idx}\n"
        f"      Fitness of best individual: {best_score}\n"
        f"      Weight of best individual: {weight}\n"
    )
