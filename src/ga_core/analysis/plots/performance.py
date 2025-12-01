"""Plot the evolution of fitness over generations."""

from matplotlib.axes import Axes

from src.ga_core.analysis.plot_context import PlotContext

PLOT_NAME = "performance"
OUTPUT_NAME = "fitness_performance_plot.png"

TITLE = "Evolution of Population Fitness and Convergence to Optimum"


def plot(ax: Axes, plot_context: PlotContext) -> None:
    """Render fitness performance plot on the provided axes.

    Args:
        ax: Matplotlib axes where the plot will be drawn.
        plot_context: Data and metadata required to render the plot.
    """
    data = plot_context.run_data
    optimum_value = plot_context.optimum

    ax.fill_between(
        data["iteration"],
        data["worst_fitness"],
        data["best_fitness"],
        label="Fitness Score range",
        color="0.9",
    )

    ax.plot(
        data["iteration"],
        data["worst_fitness"],
        label="Fitness Score of the worst individual in population",
    )

    if optimum_value is not None:
        ax.axhline(optimum_value, label=f"Known local optimum: {optimum_value}")

    ax.plot(data["iteration"], data["avg_fitness"], label="Mean fitness of population")

    ax.plot(
        data["iteration"],
        data["best_fitness"],
        label="Fitness Score of the best individual in population",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness Score")
    ax.set_title(TITLE)
