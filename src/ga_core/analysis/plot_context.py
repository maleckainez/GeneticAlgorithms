"""Shared context and specifications used by plotting utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import pandas as pd
from matplotlib.axes import Axes


@dataclass(frozen=True)
class PlotContext:
    """Input data and metadata required to render a plot."""

    run_data: pd.DataFrame
    plot_path: Path
    metadata: Optional[Mapping[str, Any]] = None
    optimum: Optional[Union[int, float]] = None


@dataclass(frozen=True)
class PlotSpec:
    """Description of a single plot type."""

    name: str
    filename: str
    func: "PlotFuncion"


PlotFuncion = Callable[[Axes, PlotContext], None]
