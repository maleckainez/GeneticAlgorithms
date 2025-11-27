"""Define timer utilities to log runtime and ETA for generations."""

from logging import LoggerAdapter
from time import perf_counter
from typing import Optional

from src.classes.ExperimentConfig import ExperimentConfig


class Timer:
    """Track generation timings and estimate time remaining."""

    def __init__(self, logger: LoggerAdapter, config: ExperimentConfig) -> None:
        """Initialize timer state and store experiment metadata.

        Args:
            logger: ``LoggerAdapter`` used for logging durations and ETA.
            config: ``ExperimentConfig`` holding total generations count.
        """
        self._start: Optional[float] = None
        self._stop: Optional[float] = None
        self._exp_start: float = perf_counter()
        self._generations: int = config.generations
        self._logger = logger
        logger.debug("TIMER initalized")

    def start(self, generation: int) -> None:
        """Mark start time for a generation.

        Args:
            generation: Current generation index (1-based).
        """
        self._start = perf_counter()
        self._stop = None
        self._logger.debug(f"Timer started at generation {generation}")

    def stop(self, generation: int) -> None:
        """Mark stop time, log elapsed, and log ETA if applicable.

        Args:
            generation: Current generation index (1-based).
        """
        if self._start is None:
            raise RuntimeError("Timer was not started")
        self._stop = perf_counter()
        self._logger.debug(f"Timer stopped for generation {generation}")
        self.elapsed(generation=generation)
        self.eta_left(generation)
        if generation == self._generations:
            runtime = self._stop - self._exp_start
            self._logger.info("Program worked for %.3f s", runtime)

    def elapsed(self, generation: int) -> None:
        """Log elapsed time for the current generation.

        Args:
            generation: Current generation index (1-based).
        """
        if self._start is None:
            raise RuntimeError("Timer was not started")
        current = self._stop if self._stop is not None else perf_counter()
        elapsed_time = current - self._start
        self._logger.info("Generation %d finished in %.3f s", generation, elapsed_time)

    def eta_left(self, generation: int) -> None:
        """Estimate remaining time and log progress percentage.

        Args:
            generation: Current generation index (1-based).
        """
        if generation <= 0:
            return
        if self._stop is None:
            total_elapsed = perf_counter() - self._exp_start
        else:
            total_elapsed = self._stop - self._exp_start
        mean_gen_time = total_elapsed / generation
        generations_left = self._generations - generation

        if generations_left <= 0:
            return

        est_time_left = generations_left * mean_gen_time
        self._logger.info(
            "Generation %d out of %d time left %.3f s | %.1f%%",
            generation,
            self._generations,
            est_time_left,
            generation * 100 / self._generations,
        )
