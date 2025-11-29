"""Tests for generation summary logging helper."""

import io
import logging

from src.ga_core.logging.generation_log import log_generation


def test_log_generation_includes_all_fields(
    clean_experiment_logger: logging.Logger,
) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("ga_core.generation.test")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    log_generation(
        logger=logger,
        best_idx=2,
        best_score=15,
        weight=7,
        iteration=3,
        repetitions=1,
    )
    handler.flush()
    content = stream.getvalue()

    assert "Generation 3" in content
    assert "Fitness of best individual: 15" in content
    assert "Weight of best individual: 7" in content
    assert "There were 1 individuals with same scores" in content


def test_log_generation_omits_repetition_line_when_zero(
    clean_experiment_logger: logging.Logger,
) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("ga_core.generation.zero")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    log_generation(
        logger=logger,
        best_idx=0,
        best_score=5,
        weight=3,
        iteration=1,
        repetitions=0,
    )
    handler.flush()
    content = stream.getvalue()

    assert "Generation 1" in content
    assert "There were" not in content
