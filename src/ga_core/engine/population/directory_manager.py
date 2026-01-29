"""Resolve and manage experiment file paths relative to a single root.

``DirectoryManager`` implements ``StorageLayout`` using simple subdirectories
(``temp``, ``output``, ``logs``, ``plots``) under a provided root. It keeps the
storage contract explicit while remaining minimal for testability.
"""

from pathlib import Path

from src.ga_core.storage.layout import StorageLayout


class DirectoryManager(StorageLayout):
    """Compute experiment directory paths under a given filesystem root.

    This class does not create directories. It only defines where temporary data,
    outputs, logs, and plots should be located relative to ``root``.
    """

    def __init__(self, root: Path) -> None:
        """Initialize resolver with a root directory for a single GA run.

        Args:
            root: Base directory for the experiment. All storage paths
                (temp/output/logs/plots) are resolved relative to this path.
        """
        self._root = root

    @property
    def temp(self) -> Path:
        """Return directory for temporary runtime-only files.

        The directory is expected to hold transient data such as memmaps and
        is safe to clean up after each run.
        """
        return self._root / "temp"

    @property
    def output(self) -> Path:
        """Return directory for final experiment outputs.

        Exported solutions, checkpoints, and artefacts intended for inspection
        belong here.
        """
        return self._root / "output"

    @property
    def logs(self) -> Path:
        """Return directory for log files produced during the run.

        The logging configuration should be applied elsewhere; the resolver
        only provides the path.
        """
        return self._root / "logs"

    @property
    def plots(self) -> Path:
        """Return directory for generated plots and visualisations.

        Physically this is stored under the output directory
        (``<root>/output/plots``), but exposed as a separate path
        for convenience.
        """
        return self.output / "plots"
