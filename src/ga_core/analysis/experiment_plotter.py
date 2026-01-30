"""Factory for creating experiment plots and rendering metadata tables."""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame

from src.ga_core.storage import StorageLayout

from .plot_context import PlotContext, PlotSpec
from .plots import performance


class ExperimentPlotter:
    """Generate plots for a single experiment run."""

    def __init__(self, plot_context: PlotContext):
        """Initialise the plotter with context and output path.

        Args:
            plot_context: Data and metadata required for plotting.
            plot_path: Destination directory for rendered plots.
        """
        self._plot_context = plot_context
        self._plot_registry: Dict[str, PlotSpec] = {
            performance.PLOT_NAME: PlotSpec(
                name=performance.PLOT_NAME,
                filename=performance.OUTPUT_NAME,
                func=performance.plot,
            ),
        }

    @classmethod
    def from_data(
        cls,
        run_data: DataFrame,
        plot_path: Path,
        metadata: Optional[Mapping[str, Any]] = None,
        optimum: Optional[Union[int, float]] = None,
    ) -> "ExperimentPlotter":
        """Create a plotter using raw data and an explicit plot directory.

        Args:
            run_data: DataFrame with logged experiment metrics.
            plot_path: Directory where plots will be saved.
            metadata: Optional metadata to render alongside plots.
            optimum: Optional known optimum value to overlay.

        Returns:
            ExperimentPlotter: Configured plotter instance.
        """
        return cls(
            PlotContext(
                run_data=run_data,
                metadata=metadata,
                optimum=optimum,
                plot_path=plot_path,
            )
        )

    @classmethod
    def from_data_and_layout(
        cls,
        run_data: DataFrame,
        layout: StorageLayout,
        metadata: Optional[Mapping[str, Any]] = None,
        optimum: Optional[Union[int, float]] = None,
    ) -> "ExperimentPlotter":
        """Create a plotter using a storage layout for plot destinations.

        Args:
            run_data: DataFrame with logged experiment metrics.
            layout: Storage layout providing plot directory.
            metadata: Optional metadata to render alongside plots.
            optimum: Optional known optimum value to overlay.

        Returns:
            ExperimentPlotter: Configured plotter instance.
        """
        return cls(
            PlotContext(
                run_data=run_data,
                metadata=metadata,
                optimum=optimum,
                plot_path=layout.plots,
            )
        )

    def run_single_plot(self, name: str) -> None:
        """Render a single plot by name and save it to disk.

        Args:
            name: Registered plot name to render.

        Raises:
            KeyError: If the plot name is not registered.
        """
        spec = self._get_plotspec(name)
        fig, ax_plot, ax_meta = self._create_figure()
        spec.func(ax_plot, self._plot_context)

        self._apply_styles(ax_plot)
        if self._plot_context.metadata is not None:
            self._render_metadata_table(ax_meta, self._plot_context.metadata)
        self._save_figure(fig, spec.filename)

    def run_select_plots(self, names: list[str]) -> None:
        """Render a subset of registered plots in sequence.

        Args:
            names: Collection of registered plot names to render.
        """
        for name in names:
            self.run_single_plot(name)

    def run_all_plots(self) -> None:
        """Render all registered plots."""
        for name in self._plot_registry:
            self.run_single_plot(name)

    def _get_plotspec(self, name: str) -> PlotSpec:
        """Return the plot specification for a given name.

        Raises:
            KeyError: If the plot name is not registered.
        """
        try:
            return self._plot_registry[name]
        except KeyError:
            raise KeyError("Unknown plot name!")

    def _create_figure(self) -> tuple[Figure, Axes, Axes]:
        """Create the figure canvas with plot and metadata subplots.

        Returns:
            tuple[Figure, Axes, Axes]: Figure with plot and metadata axes.
        """
        fig = plt.figure(figsize=(12, 6), dpi=300)
        gridspec = fig.add_gridspec(1, 2, width_ratios=[3, 1])
        ax_plot = fig.add_subplot(gridspec[0, 0])
        ax_meta = fig.add_subplot(gridspec[0, 1])
        ax_meta.axis("off")
        return fig, ax_plot, ax_meta

    def _apply_styles(self, ax: Axes) -> None:
        """Apply grid, legend, margins, and spine formatting."""
        ax.grid(axis="both", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            shadow=False,
            ncol=2,
            frameon=True,
        )
        ax.margins(0, 0.01)
        plt.subplots_adjust(
            bottom=0.25,
            right=0.95,
            top=0.90,
        )

    def _render_metadata_table(
        self, ax_meta: Axes, metadata: Mapping[str, Any]
    ) -> None:
        """Render a simple key/value table for metadata on the side axis."""
        items = [(key, value) for key, value in metadata.items() if value is not None]

        meta_table = ax_meta.table(
            cellText=[[str(key), str(value)] for key, value in items],
            colLabels=["Parameter", "Value"],
            loc="center",
        )
        meta_table.auto_set_font_size(False)
        meta_table.set_fontsize(8)
        meta_table.scale(1.0, 1.2)

    def _save_figure(self, fig: Figure, filename: str) -> None:
        """Save the rendered figure to the plot destination."""
        assert self._plot_context.plot_path.exists()
        fig.tight_layout()
        fig.savefig(self._plot_context.plot_path / filename)
