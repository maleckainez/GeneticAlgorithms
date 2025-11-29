"""High-level storage helper for a single experiment run.

This module provides a small facade around path resolution, directory creation,
and population file management so experiment runners interact with a concise,
library-friendly API. It keeps the call surface intentionally small to ease
discoverability for other modules and external tooling.
"""

from logging import Logger, LoggerAdapter
from pathlib import Path
from typing import Literal, Union

import numpy as np

from .data_paths import resolve_data_dict_path, resolve_optimum_file
from .directory_utils import cleanup_temp, ensure_layout_paths
from .layout import StorageLayout
from .path_resolver import PathResolver
from .population_files import (
    children_filepath,
    commit_children,
    create_population_file,
    load_memmap,
    population_filepath,
)

LoggerType = Union[Logger, LoggerAdapter, None]


class ExperimentStorage:
    """Wrap storage layout, data resolution, and memmap paths for one run.

    Instances keep track of an experiment's root layout, the population file
    base name, and the source data identifier. Methods return resolved paths or
    delegate to specialised helpers to perform safe file moves and cleanups.
    """

    def __init__(
        self,
        layout: StorageLayout,
        filename: str,
        data_file_name: str,
        logger: LoggerType = None,
    ) -> None:
        """Create storage helper with pre-resolved names and optional logger.

        Args:
            layout: Concrete ``StorageLayout`` implementation for this run.
            filename: Base name used for population and children files (without
                extension).
            data_file_name: Filename of the input dataset used by the
                experiment.
            logger: Optional logger or adapter for informational or debug
                messages.
        """
        self._layout = layout
        self._file_name = filename
        self._data_file_name = data_file_name
        self._logger = logger
        if self._logger is not None:
            self._logger.debug("ExperimentStorage initialized successfully.")

    @classmethod
    def from_root(
        cls, root: Path, filename: str, data_file_name: str, logger: LoggerType = None
    ) -> "ExperimentStorage":
        """Create storage at ``root`` and ensure required directories exist.

        Use this constructor when starting a new experiment. It prepares all
        directories needed to store populations, logs, outputs, and temporary
        data under ``root``.

        Args:
            root: Base directory for the experiment's artefacts.
            filename: Base name used for population and children files (without
                extension).
            data_file_name: File stem of the input dataset used by the
                experiment.
            logger: Optional logger or adapter for informational or debug
                messages.

        Returns:
            ExperimentStorage: Initialised storage helper ready for use.
        """
        layout = PathResolver(root)
        ensure_layout_paths(layout)
        if logger is not None:
            logger.info(
                "Initialized storage at root %s (experiment file: %s, data file: %s)",
                root,
                filename,
                data_file_name,
            )
        return cls(layout, filename, data_file_name)

    def children_path(self) -> Path:
        """Return path to the temporary children memmap file.

        Returns:
            Path: Resolved path for the children population file stored
                in the temporary directory.
        """
        return children_filepath(layout=self._layout, file_name=self._file_name)

    def commit_children(self, expected_size: int, retries: int = 3) -> None:
        """Atomically replace population memmap with children memmap.

        Args:
            expected_size: Required size of the children memmap in bytes. The
                file is validated before replacement.
            retries: Number of attempts to perform the atomic swap before
                raising an error.
        """
        commit_children(
            layout=self._layout,
            file_name=self._file_name,
            expected_size=expected_size,
            retries=retries,
            logger=self._logger,
        )

    def population_path(self) -> Path:
        """Return path to the main population memmap file.

        Returns:
            Path: Resolved path for the main population memmap file stored in
                the temporary directory.
        """
        return population_filepath(layout=self._layout, file_name=self._file_name)

    def data_dict_path(self) -> Path:
        """Return path to the input data file for this experiment.

        Returns:
            Path: Absolute path to the selected input dataset file.
        """
        return resolve_data_dict_path(self._data_file_name)

    def optimum_file(self) -> Path:
        """Return file containing optimum values for this instance data file.

        Returns:
            Path: File containing optimum solutions for the specific data file.
        """
        return resolve_optimum_file(self._data_file_name)

    def remove_temp_data(self) -> None:
        """Remove temporary storage directory and its contents."""
        cleanup_temp(self._layout)
        if self._logger is not None:
            self._logger.info(
                "Removed experiment's temporary directory and files %s",
                self._layout.temp,
            )

    def ensure_storage_exists(self) -> None:
        """Ensure all layout directories exist on disk.

        Creates the directories described by the layout.
        """
        ensure_layout_paths(self._layout)
        if self._logger is not None:
            self._logger.debug(
                "Ensured storage directories exist for root %s",
                self._layout.output.parent,
            )

    def create_population_memmap(
        self,
        population_size: int,
        genome_length: int,
        data_type: type = np.uint8,
    ) -> None:
        """Create an empty population memmap file for this experiment.

        The memmap and its JSON configuration are created in the temporary
        directory configured by the storage layout.

        Args:
            population_size: Number of individuals in the population.
            genome_length: Number of genes in a single genome.
            data_type: NumPy-compatible data type used to store the population.
        """
        create_population_file(
            population_size=population_size,
            genome_length=genome_length,
            temp=self._layout.temp,
            filename=self._file_name,
            data_type=data_type,
        )

    def load_population_memmap(
        self,
        open_mode: Literal[
            "readonly", "r", "copyonwrite", "c", "readwrite", "r+", "write", "w+"
        ] = "r",
    ) -> np.memmap:
        """Load the population memmap associated with this experiment.

        The method validates the on-disk configuration before opening the
        memmap file.

        Args:
            open_mode: NumPy memmap open mode, for example ``"r"`` or
                ``"r+"``.

        Returns:
            np.memmap: Memory-mapped array representing the population.
        """
        population, _ = load_memmap(
            temp=self._layout.temp,
            filename=self._file_name,
            open_mode=open_mode,
        )
        return population
