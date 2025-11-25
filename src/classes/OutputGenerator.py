"""Module for managing CSV file output for experiment results."""

import csv
from pathlib import Path
from typing import Any, Optional, TextIO

from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver


class OutputGenerator:
    """Manages the creation and writing of experiment data to a CSV file.

    Handles opening the file, writing initial metadata (configuration),
    appending iteration results, and ensuring safe closure of the file handle.
    """

    def __init__(self, pr: PathResolver, config: ExperimentConfig) -> None:
        """Initializes the generator and determines the output file path.

        Args:
            pr (PathResolver): PathResolver instance providing output directory.
            config (ExperimentConfig): Configuration object with experiment details.
        """
        self.config = config
        self.pr = pr
        self.plot_path = pr.get_plot_path()
        self.input_path = pr.get_output_path()
        self.filename = Path(self.input_path / f"{pr.filename_constant}.csv")
        self.file: Optional[TextIO] = None
        self.writer: Optional[Any] = None

    def _open(self) -> None:
        """Opens the CSV file for writing and initializes the CSV writer object."""
        if self.file is None:
            self.file = open(self.filename, "w", newline="")
            self.writer = csv.writer(self.file)

    def close(self) -> None:
        """Opens the CSV file for writing and initializes the CSV writer object."""
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None

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
            iteration (int): Current generation number.
            best_fitness (int): Fitness of the best individual.
            best_weight (int): Weight of the best individual's knapsack.
            avg_fitness (float): Average fitness of the population.
            worst_fitness (int): Fitness of the worst individual.
            worst_weight (int): Weight of the worst individual's knapsack.
            identical_best_count (int): Number of individuals with the best fitness.
            genome (str): Genome string of the best individual.

        Raises:
            RuntimeError: If the file has not been opened (i.e., `_open()`
                          or `init_csv()` was not called).
        """
        if self.writer is None:
            raise RuntimeError("Plotter not opened. Call .open() or .init_csv() first.")
        self.writer.writerow(
            [
                iteration,
                best_fitness,
                best_weight,
                avg_fitness,
                worst_fitness,
                worst_weight,
                identical_best_count,
                genome,
            ]
        )

    def init_csv(self, config: ExperimentConfig) -> None:
        """Initializes the CSV file by writing configuration metadata and header row.

        The metadata lines are prefixed with `#` so they can be ignored during
        data loading for plotting.

        Args:
            config (ExperimentConfig): The configuration object used to extract
                                       metadata fields.
        """
        if self.file is None or self.writer is None:
            self._open()
        assert self.file is not None and self.writer is not None
        meta_rows = [
            ["# data_filename", config.data_filename],
            ["# population_size", config.population_size],
            ["# generations", config.generations],
            ["# max_weight", config.max_weight],
            ["# seed", config.seed],
            ["# selection_type", config.selection_type],
            ["# crossover_type", config.crossover_type],
            ["# crossover_probability", config.crossover_probability],
            ["# mutation_probability", config.mutation_probability],
            ["# penalty", config.penalty],
            ["# experiment_identifier", config.experiment_identifier],
            ["# log_level", config.log_level],
            ["# stream_batch_size", config.stream_batch_size],
            ["# selection_pressure", config.selection_pressure],
            [],
        ]

        header = [
            "iteration",
            "best_fitness",
            "best_weight",
            "avg_fitness",
            "worst_fitness",
            "worst_weight",
            "identical_best_individuals_repetitions",
            "genome_of_best_individual",
        ]

        self.writer.writerows(meta_rows)
        self.writer.writerow(header)
