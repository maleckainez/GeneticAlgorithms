import matplotlib.pyplot as plt
import pandas

from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver
from pathlib import Path
import csv

class Plotter:

    def __init__(self,
                 pr: PathResolver,
                 config = ExperimentConfig):
        self.config = config
        self.pr = pr
        self.plot_path = pr.get_plot_path()
        self.input_path = pr.get_output_path()
        self.filename = Path( self.input_path / f"{pr.filename_constant}.csv")
        self.file = None
        self.writer = None

    def _open(self):
        if self.file is None:
            self.file = open(self.filename, 'w', newline='')
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
            "identical_best_count",
            "genome_of_best_individual"
        ]

        self.writer.writerows(meta_rows)
        self.writer.writerow(header)

    def _load_experiment_data(self):
        data_file = pandas.read_csv(self.filename, comment='#')
        return data_file

    def _load_optimum_data(self):
        optimum_dir = self.pr.get_optimum_path()
        optimum_filepath = Path(optimum_dir) / self.config.data_filename
        datafile = pandas.read_csv(optimum_filepath, header= None)
        optimum_value = datafile.iloc[0,0]
        return optimum_value

    def best_fitness_plot(self):
        datafile = self._load_experiment_data()

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

        ax.plot(
            datafile["iteration"],
            datafile["best_fitness"],
            label="Fitness of best individual",
            linewidth=2,
            marker="o",
            markersize=3,
        )
        ax.plot(
            datafile["iteration"],
            datafile["avg_fitness"],
            label="Average fitness of population",
            linewidth=2,
            linestyle="--",
            marker="s",
            markersize=3,
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness")
        ax.set_title("Best and average fitness per iteration")

        ax.grid(axis="both", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        legend = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            shadow=False,
            ncol=2,
            frameon=True,
        )

        fig.tight_layout()
        plt.subplots_adjust(bottom=0.25)

        fig.savefig(Path(self.plot_path)/"best_and_avg_pop_fitness.png")
        plt.close(fig)

    def best_fitness_v_optimum(self):
        datafile = self._load_experiment_data()

        optimum = self._load_optimum_data()

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

        ax.plot(
            datafile["iteration"],
            datafile["best_fitness"],
            label="Fitness of best individual",
            linewidth=2,
            marker="o",
            markersize=3,
        )
        ax.axhline(optimum,
                   label=f"Optimum value ({optimum})",
                   linewidth=2,
                   linestyle="--",
                   color= 'r',
                   marker=None,
                   markersize=0,)


        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness")
        ax.set_title("Best and average fitness per iteration")

        ax.grid(axis="both", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        legend = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            shadow=False,
            ncol=2,
            frameon=True,
        )

        fig.tight_layout()
        plt.subplots_adjust(bottom=0.25)

        fig.savefig(Path(self.plot_path)/"best_fitness_v_optimal.png")
        plt.close(fig)
