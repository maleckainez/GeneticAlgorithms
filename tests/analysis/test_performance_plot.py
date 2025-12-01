"""Tests for the performance plot helper."""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from src.ga_core.analysis.plot_context import PlotContext
from src.ga_core.analysis.plots.performance import plot

matplotlib.use("Agg")


def test_performance_plot_renders_lines(tmp_path) -> None:
    data = pd.DataFrame(
        {
            "iteration": [0, 1, 2],
            "worst_fitness": [0, 1, 2],
            "avg_fitness": [0.5, 1.0, 1.5],
            "best_fitness": [1, 2, 3],
        }
    )
    ctx = PlotContext(run_data=data, metadata=None, optimum=2, plot_path=tmp_path)
    fig, ax = plt.subplots()

    plot(ax, ctx)

    # worst_fitness + avg_fitness + best_fitness + optimum line = 4 artists
    assert len(ax.lines) >= 4
