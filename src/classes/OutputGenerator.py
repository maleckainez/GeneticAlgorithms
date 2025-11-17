from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver
from pathlib import Path
import csv


class OutputGenerator:

    def __init__(self, pr: PathResolver, config: ExperimentConfig):
        self.config = config
        self.pr = pr
        self.plot_path = pr.get_plot_path()
        self.input_path = pr.get_output_path()
        self.filename = Path(self.input_path / f"{pr.filename_constant}.csv")
        self.file = None
        self.writer = None

    def _open(self):
        if self.file is None:
            self.file = open(self.filename, "w", newline="")
            self.writer = csv.writer(self.file)

    def close(self):
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
    ):
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

    def init_csv(self, config: ExperimentConfig):
        if self.file is None:
            self._open()
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
