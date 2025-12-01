"""CSV helpers used to save experiment results.

These writers wrap :class:`~src.ga_core.output.csv_handler.CsvHandler` to add
simple metadata rows and to check that every data row matches the header.
They expect that the output directory already exists.
"""

from pathlib import Path
from types import TracebackType
from typing import Any, Mapping, Optional

from src.ga_core.config import ExperimentConfig
from src.ga_core.config.input_config_scheme import (
    CrossoverType,
    LogLevel,
    SelectionType,
)
from src.ga_core.storage import StorageLayout

from .csv_handler import CsvHandler


class CsvGenericOutput:
    """Generic CSV writer with optional metadata and header checks.

    It can write metadata rows formatted as ``#<key>,<value>``, then headers,
    and later data rows that must match the header length.
    """

    def __init__(self, handler: CsvHandler) -> None:
        """Initialize the CSV writer with a concrete handler.

        Args:
            handler: Low-level CSV file handler.
        """
        self.handler = handler
        self._headers_amount: Optional[int] = None

    @classmethod
    def from_layout(
        cls, exp_filename: str, layout: StorageLayout
    ) -> "CsvGenericOutput":
        """Create a CSV output using a layout and explicit filename.

        Args:
            exp_filename: Base filename without extension.
            layout: Storage layout providing the output directory.

        Returns:
            CsvGenericOutput: Bound to ``<layout.output>/<exp_filename>.csv``.
        """
        handler = CsvHandler(output_path=layout.output, exp_filename=exp_filename)
        return cls(handler)

    @classmethod
    def from_input(cls, exp_filename: str, output_path: Path) -> "CsvGenericOutput":
        """Create a CSV output using an explicit filename and output path.

        Args:
            exp_filename: Base filename without extension.
            output_path: Directory where the CSV will be written.

        Returns:
            CsvGenericOutput: Bound to
            ``<output_path>/<exp_filename>.csv``.
        """
        handler = CsvHandler(output_path, exp_filename)
        return cls(handler)

    def __enter__(self) -> "CsvGenericOutput":
        """Open the underlying handler when entering a context."""
        self.handler.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None:
        """Close the underlying handler when exiting a context."""
        self.handler.close()

    @property
    def writer(self) -> Any:
        """Return the active CSV writer."""
        return self.handler.writer

    def init_csv(
        self, headers: list[str], metadata: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Write optional metadata rows followed by column headers.

        Metadata rows use ``#<key>`` in the first column and the value in the
        second column. A blank line and the header row follow.
        """
        if metadata:
            meta_rows = [
                [f"#{meta_name}", meta_data]
                for meta_name, meta_data in metadata.items()
            ]
            meta_rows.append([])
            self.writer.writerows(meta_rows)
        self.writer.writerow(headers)
        self._headers_amount = len(headers)

    def write_csv_row(self, *vals: Any) -> None:
        """Write one data row and check it matches the header length.

        Raises:
            RuntimeError: If headers were not initialised first.
            ValueError: If the provided values do not match header count.
        """
        if self._headers_amount is None:
            raise RuntimeError("CSV was not initialized. Run init_csv() first!")
        if len(vals) != self._headers_amount:
            raise ValueError(
                f"{len(vals)} values, but header has {self._headers_amount} columns."
            )
        self.writer.writerow(vals)

    def __call__(self, *values: Any) -> None:
        """Proxy call syntax to ``write_csv_row``."""
        self.write_csv_row(*values)


class ExperimentCsvOutput(CsvGenericOutput):
    """CSV writer for standard genetic algorithm metrics.

    Uses the column order from :data:`HEADERS` and checks that selection
    metadata for rank or tournament strategies is provided when needed.
    """

    HEADERS = [
        "iteration",
        "best_fitness",
        "best_weight",
        "avg_fitness",
        "worst_fitness",
        "worst_weight",
        "identical_best_individuals_repetitions",
        "genome_of_best_individual",
    ]

    @classmethod
    def from_config_and_layout(
        cls, config: ExperimentConfig, layout: StorageLayout
    ) -> "CsvGenericOutput":
        """Create a CSV output using experiment config and storage layout.

        Args:
            config: Experiment configuration containing the job identifier.
            layout: Storage layout providing the output directory.

        Returns:
            CsvGenericOutput: Bound to ``<layout.output>/<job_id>.csv``.
        """
        handler = CsvHandler(output_path=layout.output, exp_filename=config.job_id)
        return cls(handler)

    @classmethod
    def from_config(
        cls, config: ExperimentConfig, output_path: Path
    ) -> "CsvGenericOutput":
        """Create a CSV output using config and an explicit path.

        Args:
            config: Experiment configuration containing the job identifier.
            output_path: Directory where the CSV will be written.

        Returns:
            CsvGenericOutput: Bound to ``<output_path>/<job_id>.csv``.
        """
        handler = CsvHandler(output_path, exp_filename=config.job_id)
        return cls(handler)

    def init_experiment_csv(
        self,
        *,
        job_id: str,
        data_filename: str,
        population_size: int,
        generations: int,
        max_weight: int,
        seed: Optional[int] = None,
        selection_type: str,
        crossover_type: str,
        crossover_probability: float,
        mutation_probability: float,
        penalty: float,
        experiment_identifier: Optional[str] = None,
        log_level: Optional[str] = None,
        selection_pressure: Optional[float] = None,
        tournament_size: Optional[int] = None,
    ) -> None:
        """Initialize CSV with experiment metadata and standard headers.

        Args:
            job_id: Unique experiment identifier.
            data_filename: Dataset filename used for the run.
            population_size: Population size.
            generations: Number of generations.
            max_weight: Maximum knapsack weight.
            seed: RNG seed used for reproducibility.
            selection_type: Selection strategy name.
            crossover_type: Crossover operator name.
            crossover_probability: Probability of crossover.
            mutation_probability: Probability of mutation.
            penalty: Penalty multiplier applied to overweight solutions.
            experiment_identifier: Human-friendly experiment tag.
            log_level: Logging level string.
            selection_pressure: Rank selection pressure, when applicable.
            tournament_size: Tournament size, when applicable.

        Raises:
            ValueError: If selection-specific parameters are missing.
        """
        metadata: dict[str, Any] = {
            "job_id": job_id,
            "experiment_identifier": experiment_identifier,
            "data_filename": data_filename,
            "population_size": population_size,
            "generations": generations,
            "max_weight": max_weight,
            "seed": seed,
            "selection_type": selection_type,
            "crossover_type": crossover_type,
            "crossover_probability": crossover_probability,
            "mutation_probability": mutation_probability,
            "penalty": penalty,
            "log_level": log_level,
        }
        if selection_type == SelectionType.TOURNAMENT.value:
            if tournament_size is None:
                raise ValueError("Tournament size value missing.")
            metadata["tournament_size"] = tournament_size

        if selection_type == SelectionType.LINEAR_RANK.value:
            if selection_pressure is None:
                raise ValueError("Selection pressure value missing.")
            metadata["selection_pressure"] = selection_pressure

        self.init_csv(headers=self.HEADERS, metadata=metadata)

    def init_experiment_csv_from_config(self, config: ExperimentConfig) -> None:
        """Populate metadata from an ExperimentConfig instance."""
        selection_type: SelectionType = config.selection_type
        crossover_type: CrossoverType = config.crossover_type
        log_level: LogLevel = config.log_level
        self.init_experiment_csv(
            job_id=config.job_id,
            data_filename=config.data_filename,
            population_size=config.population_size,
            generations=config.generations,
            max_weight=config.max_weight,
            seed=config.seed,
            selection_type=selection_type,
            crossover_type=crossover_type.value,
            crossover_probability=config.crossover_probability,
            mutation_probability=config.mutation_probability,
            penalty=config.penalty_multiplier,
            experiment_identifier=config.experiment_identifier,
            log_level=log_level.value,
            selection_pressure=config.selection_pressure,
            tournament_size=config.tournament_size,
        )

    def write_iteration(
        self,
        iteration: int,
        best_fitness: int,
        best_weight: int,
        avg_fitness: float,
        worst_fitness: int,
        worst_weight: int,
        identical_best_count: int,
        genome: str,
    ) -> None:
        """Appends a single row of metrics from one generation to the CSV file.

        Args:
            iteration: Current generation number.
            best_fitness: Fitness of the best individual.
            best_weight: Weight of the best individual's knapsack.
            avg_fitness: Average fitness of the population.
            worst_fitness: Fitness of the worst individual.
            worst_weight: Weight of the worst individual's knapsack.
            identical_best_count: Number of individuals with the best fitness.
            genome: Genome string of the best individual.

        Raises:
            RuntimeError: If the file has not been opened or initialized.
        """
        vals = (
            iteration,
            best_fitness,
            best_weight,
            avg_fitness,
            worst_fitness,
            worst_weight,
            identical_best_count,
            genome,
        )
        self.write_csv_row(*vals)
