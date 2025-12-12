"""Timer utilities for logging per-generation runtime and ETA.

This module keeps timing concerns isolated and side-effect free aside from
optional logging. If you skip the total generation count, the timer still
measures elapsed time but will not compute ETA. Callers configure logging
handlers; the timer only writes to a provided logger/adapter when present.
"""

from logging import Logger, LoggerAdapter
from time import perf_counter
from typing import Optional, Union

from src.ga_core.config.experiment_config import ExperimentConfig
from src.ga_core.config.input_config_scheme import InputConfig

ConfigLike = Union[int, ExperimentConfig, InputConfig, None]
LoggerType = Union[Logger, LoggerAdapter, None]


class Timer:
    """Track per-generation timings and estimate remaining time.

    The timer is reusable across runners: it optionally accepts a raw generation
    count or a config object exposing ``generations`` and, optionally,
    a logger/adapter. No logging handlers are created here. Calling ``start``
    again resets the timer even if ``stop`` was not called.

    Typical usage::

        timer = Timer.from_generations(100, logger=my_logger)
        for gen in range(1, 101):
            timer.start(gen)
            run_generation(gen)
            elapsed, eta = timer.stop(gen)

    The returned ``elapsed`` and ``eta`` values are expressed in seconds.
    """

    def __init__(
        self, config_like: ConfigLike = None, logger: LoggerType = None
    ) -> None:
        """Initialize timer state and record total generations count.

        Args:
            config_like: Optional number of generations or an object
                exposing it (``ExperimentConfig`` or ``JobConfig``). When ``None``,
                elapsed time is available but ETA stays disabled.
            logger: Optional logger/adapter used for debug/INFO timings.
        """
        self._start: Optional[float] = None
        self._stop: Optional[float] = None
        self._exp_start: float = perf_counter()
        self._total_generations: Optional[int] = self._normalize_config(config_like)
        self._logger = logger
        if self._logger is not None:
            self._logger.debug("TIMER initialized")

    @classmethod
    def with_int_generations(
        cls, total_generations: int, logger: LoggerType = None
    ) -> "Timer":
        """Create a timer from a raw generation count.

        Args:
            total_generations: Total number of generations to track.
            logger: Optional logger or logger adapter.

        Returns:
            Timer: Configured timer instance.
        """
        return cls(config_like=total_generations, logger=logger)

    @classmethod
    def with_experiment_config_class(
        cls, config: ExperimentConfig, logger: LoggerType = None
    ) -> "Timer":
        """Create a timer from an ``ExperimentConfig`` instance."""
        return cls(config_like=config, logger=logger)

    @classmethod
    def with_job_config(
        cls, job_config: InputConfig, logger: LoggerType = None
    ) -> "Timer":
        """Create a timer from a validated ``InputConfig`` instance."""
        return cls(config_like=job_config, logger=logger)

    @classmethod
    def with_no_config(
        cls, job_config: None = None, logger: LoggerType = None
    ) -> "Timer":
        """Create a timer without a total generation count (no ETA)."""
        return cls(config_like=None, logger=logger)

    def _normalize_config(self, config_like: ConfigLike) -> Optional[int]:
        """Resolve the total generations count from supported inputs.

        Args:
            config_like: Integer, ``ExperimentConfig``, ``JobConfig``, or None.

        Returns:
            int | None: Total number of generations, or ``None`` to disable ETA.

        Raises:
            TypeError: If the provided object does not expose a generation count
                and is not None.
        """
        if isinstance(config_like, int):
            return config_like
        if isinstance(config_like, ExperimentConfig):
            return config_like.generations
        if isinstance(config_like, InputConfig):
            return config_like.population.generations
        if config_like is None:
            return None
        raise TypeError(
            f"Unsupported config type: {type(config_like)!r}",
        )

    def start(self, generation: int) -> None:
        """Mark start time for a generation.

        Args:
            generation: Current generation index (must be positive).
        """
        self._start = perf_counter()
        self._stop = None
        if self._logger is not None:
            self._logger.debug("Timer started at generation %d", generation)

    def stop(self, generation: int) -> tuple[float, Optional[float]]:
        """Mark stop time, log elapsed, and log ETA if applicable.

        Args:
            generation: Current generation index (must be positive).

        Returns:
            tuple[float, Optional[float]]: Elapsed seconds for the generation and
                optional estimated time left (seconds). The ETA element can be ``None``
                when the generation index is invalid, not positive, total generations
                were not defined, or the GA algorithm is already complete.

        Raises:
            RuntimeError: If the timer was not started.
        """
        if self._start is None:
            raise RuntimeError("Timer was not started")
        self._stop = perf_counter()
        elapsed_time = self.elapsed()

        if self._logger is not None:
            self._logger.debug("Timer stopped for generation %d", generation)
            self._logger.info(
                "Generation %d finished in %.3f s", generation, elapsed_time
            )

        total_eta = self.eta_left(generation)

        if generation == self._total_generations and self._logger is not None:
            total_runtime = self._stop - self._exp_start
            self._logger.info("Program worked for %.3f s", total_runtime)

        return (elapsed_time, total_eta)

    def elapsed(self) -> float:
        """Return elapsed time for the current generation in seconds.

        Raises:
            RuntimeError: If the timer was not started.
        """
        if self._start is None:
            raise RuntimeError("Timer was not started")
        current = self._stop if self._stop is not None else perf_counter()
        elapsed_time = current - self._start
        return elapsed_time

    def eta_left(self, generation: int) -> Optional[float]:
        """Estimate remaining time and log progress percentage.

        Args:
            generation: Current generation index (must be positive).

        Returns:
            Optional[float]: Estimated seconds remaining, or ``None`` when
                total number of generations was not provided, present generation
                is non-positive or already finished.
        """
        if self._total_generations is None:
            return None
        if generation <= 0:
            return None
        if self._stop is None:
            total_elapsed = perf_counter() - self._exp_start
        else:
            total_elapsed = self._stop - self._exp_start

        mean_gen_time = total_elapsed / generation
        generations_left = self._total_generations - generation

        if generations_left <= 0:
            return None

        est_time_left = generations_left * mean_gen_time
        if self._logger is not None:
            self._logger.info(
                "Generation %d out of %d time left %.3f s | %.1f%%",
                generation,
                self._total_generations,
                est_time_left,
                generation * 100 / self._total_generations,
            )
        return est_time_left
