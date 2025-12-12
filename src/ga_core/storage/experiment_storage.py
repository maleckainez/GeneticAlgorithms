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
from .naming_scheme import NAMING, FileNamingScheme
from .population_files import (
    commit_children,
    create_empty_memmap,
    load_memmap,
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
        job_id: str,
        data_file_name: str,
        naming: FileNamingScheme = NAMING,
        logger: LoggerType = None,
    ) -> None:
        """Create storage helper with pre-resolved names and optional logger.

        Args:
            layout: Concrete ``StorageLayout`` implementation for this run.
            job_id: Base name used for population and children files (without
                extension).
            data_file_name: Filename of the input dataset used by the
                experiment.
            naming: Strategy that defines file naming conventions.
            logger: Optional logger or adapter for informational or debug
                messages.
        """
        self._layout = layout
        self._file_name = job_id
        self._data_file_name = data_file_name
        self._naming = naming
        self._logger = logger
        if self._logger is not None:
            self._logger.debug("ExperimentStorage initialized successfully.")

    def children_name(self) -> str:
        """Return string name of the children memmap file."""
        return self._naming.children_file(self._file_name)

    def population_name(self) -> str:
        """Return string name of the population memmap file."""
        return self._naming.population_file(self._file_name)

    def commit_children(self, expected_size: int, retries: int = 3) -> None:
        """Atomically replace population memmap with children memmap.

        Args:
            expected_size: Required size of the children memmap in bytes. The
                file is validated before replacement.
            retries: Number of attempts to perform the atomic swap before
                raising an error.
        """
        commit_children(
            temp_path=self._layout.temp,
            file_name=self._file_name,
            expected_size=expected_size,
            retries=retries,
            logger=self._logger,
        )

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

    def create_empty_memmap(
        self,
        population_size: int,
        genome_length: int,
        data_type: type = np.uint8,
    ) -> None:
        """Create an empty population memmap file for this experiment.

        The memmap and its JSON configuration are created in the temporary
        directory configured by the storage layout.

        Args:
            filename: Base name used for the memmap file (no extension).
            population_size: Number of individuals in the population.
            genome_length: Number of genes in a single genome.
            data_type: NumPy-compatible data type used to store the population.
        """
        create_empty_memmap(
            population_size=population_size,
            genome_length=genome_length,
            temp_path=self._layout.temp,
            filename=self._data_file_name,
            data_type=data_type,
        )

    def load_memmap(
        self,
        filename: str,
        open_mode: Literal[
            "readonly", "r", "copyonwrite", "c", "readwrite", "r+", "write", "w+"
        ] = "r",
    ) -> np.memmap:
        """Load the population memmap associated with this experiment.

        The method validates the on-disk configuration before opening the
        memmap file.

        Args:
            filename: Base name used for the memmap file (no extension).
            open_mode: NumPy memmap open mode, for example ``"r"`` or
                ``"r+"``.

        Returns:
            np.memmap: Memory-mapped array representing the population.
        """
        population, _ = load_memmap(
            temp=self._layout.temp,
            filename=filename,
            open_mode=open_mode,
        )
        return population
