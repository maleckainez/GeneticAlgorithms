"""Interfaces describing directory layouts for GA runs.

The ``StorageLayout`` protocol defines the small contract required by storage
helpers: four paths that encapsulate where temporary data, final outputs, logs,
and plots live for a single experiment. Implementations can map these to any
directory structure as long as they respect the contract.
"""

from pathlib import Path
from typing import Protocol


class StorageLayout(Protocol):
    """Describe the directory layout for a single GA run.

    Implementations provide paths used by the engine: temporary files, final
    outputs, logs, and plots. The protocol deliberately omits methods to keep
    the API narrow and focused on filesystem locations.
    """

    @property
    def temp(self) -> Path:
        """Return path to the directory for temporary files.

        Temporary files can include intermediate populations or memmaps that do
        not need to be persisted after the run finishes.
        """

    @property
    def output(self) -> Path:
        """Return path to the directory for final output files.

        This directory should be suitable for artefacts shared beyond the
        current run, such as checkpoints or exported solutions.
        """

    @property
    def logs(self) -> Path:
        """Return path to the directory for runtime log files.

        Logging configuration is handled elsewhere; this property simply names
        the destination directory.
        """

    @property
    def plots(self) -> Path:
        """Return path to the directory for generated plots.

        Plot files are typically derived from the output data and are kept
        separate for clarity.
        """
