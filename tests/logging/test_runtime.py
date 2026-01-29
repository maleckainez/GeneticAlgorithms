"""Tests for logger initialization helpers."""

import logging
from pathlib import Path

from src.ga_core.logging.runtime import (
    from_config,
    from_config_and_layout,
    from_layout,
    initialize,
)


def _flush_handlers(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        if hasattr(handler, "flush"):
            handler.flush()


def test_initialize_creates_handlers_and_log_file(
    tmp_path: Path, clean_experiment_logger: logging.Logger
) -> None:
    adapter = initialize(log_level="INFO", log_path=tmp_path, file_name="job1")

    assert adapter.logger.getEffectiveLevel() == logging.INFO
    assert len(adapter.logger.handlers) == 2

    adapter.info("hello logger")
    _flush_handlers(adapter.logger)
    log_file = tmp_path / "runtime_experiment_job1.log"
    assert log_file.exists()
    assert "hello logger" in log_file.read_text()


def test_initialize_sets_invalid_level_to_critical(
    tmp_path: Path, clean_experiment_logger: logging.Logger
) -> None:
    logger = initialize(log_level="INVALID", log_path=tmp_path, file_name="job1")
    assert logger.getEffectiveLevel() == logging.CRITICAL


def test_initialize_does_not_duplicate_handlers(
    tmp_path: Path, clean_experiment_logger: logging.Logger
) -> None:
    first = initialize(log_level="INFO", log_path=tmp_path, file_name="job1")
    handler_ids = {id(h) for h in first.logger.handlers}

    second = initialize(log_level="INFO", log_path=tmp_path, file_name="job1")
    assert {id(h) for h in second.logger.handlers} == handler_ids


def test_from_config_and_layout_writes_file(
    experiment_config,
    storage_layout,
    clean_experiment_logger: logging.Logger,
) -> None:
    adapter = from_config_and_layout(config=experiment_config, layout=storage_layout)
    adapter.info("message from config+layout")
    _flush_handlers(adapter.logger)

    log_file = storage_layout.logs / "runtime_experiment_job-123.log"
    assert log_file.exists()
    assert "message from config+layout" in log_file.read_text()


def test_from_config_uses_provided_log_path(
    experiment_config,
    tmp_path: Path,
    clean_experiment_logger: logging.Logger,
) -> None:
    adapter = from_config(config=experiment_config, log_path=tmp_path)
    adapter.info("message from config")
    _flush_handlers(adapter.logger)

    log_file = tmp_path / "runtime_experiment_job-123.log"
    assert log_file.exists()
    assert "message from config" in log_file.read_text()


def test_from_layout_uses_explicit_parameters(
    storage_layout,
    clean_experiment_logger: logging.Logger,
) -> None:
    adapter = from_layout(log_level="DEBUG", layout=storage_layout, file_name="custom")
    adapter.debug("message from layout")
    _flush_handlers(adapter.logger)

    log_file = storage_layout.logs / "runtime_experiment_custom.log"
    assert log_file.exists()
    assert "message from layout" in log_file.read_text()
