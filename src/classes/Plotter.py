from src.classes.PathResolver import PathResolver
from src.classes.ExperimentConfig import ExperimentConfig
from pathlib import Path
import pandas
from matplotlib import pyplot as plt

class Plotter:

    def __init__(self,
                 pr: PathResolver,
                 config: ExperimentConfig):
        self.config = config
        self.pr = pr
        self.plot_path = pr.get_plot_path()
        self.input_path = pr.get_output_path()
        self.filename = Path( self.input_path / f"{pr.filename_constant}.csv")
        self.file = None
        self.writer = None

    def _load_experiment_data(self):
        data_file = pandas.read_csv(self.filename, comment="#")
        return data_file

    def _load_optimum_data(self):
        optimum_dir = self.pr.get_optimum_path()
        optimum_filepath = Path(optimum_dir) / self.config.data_filename
        datafile = pandas.read_csv(optimum_filepath, header=None)
        optimum_value = datafile.iloc[0, 0]
        return optimum_value


    def performance_and_correctness(self):
        optimum_data = self._load_optimum_data()
        datafile = self._load_experiment_data()
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

        ax.fill_between(
            datafile["iteration"],
            datafile["worst_fitness"],
            datafile["best_fitness"],
            label="Fitness score range",
            color="0.9",
        )
        ax.plot(
            datafile["iteration"],
            datafile["worst_fitness"],
            label="Fitness of the worst individual",
            color='mediumpurple',
            linewidth=2,
            marker='.',
            markersize=1,

        )
        ax.axhline(
            optimum_data,
            label=f"Optimum value ({optimum_data})",
            linewidth=2,
            linestyle="--",
            color="r",
            marker=None,
            markersize=0,
        )
        ax.plot(
            datafile["iteration"],
            datafile["avg_fitness"],
            label="Average fitness of population",
            linewidth=2,
            linestyle="-",
            marker="s",
            markersize=2,
            color="C1"
        )
        ax.plot(
            datafile["iteration"],
            datafile["best_fitness"],
            label="Fitness of best individual",
            linewidth=2,
            marker="o",
            markersize=3,
            color="C0"

        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness")
        ax.set_title("Evolution of Population Fitness and Convergence to Optimum")

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
        ax.margins(0,0.01)
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.25, right=0.95, top=0.90, )

        fig.savefig(Path(self.plot_path) / "best_fitness_v_optimal.png")
        plt.close(fig)


