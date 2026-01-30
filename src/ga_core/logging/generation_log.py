"""Utilities for logging generation-level summaries."""

from logging import Logger, LoggerAdapter


def log_generation(
    logger: LoggerAdapter | Logger,
    best_idx: int,
    best_score: int,
    weight: int,
    iteration: int,
    repetitions: int,
) -> None:
    """Log summary information for a single generation.

    Args:
        logger: Logger or adapter used for writing the message.
        best_idx: Index of the best individual in the population.
        best_score: Fitness score of the best individual.
        weight: Weight of the best individual.
        iteration: Current generation number.
        repetitions: Count of individuals sharing the best score.
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
