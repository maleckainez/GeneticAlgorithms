"""Module for creation and management of memory-mapped population file.

It provides the PopulationHandler class, which is responsible for creating,
accessing, and safely closing the NumPy memmap file used to temporarily store
the population during the reproduction cycle.
"""

from typing import Literal, Optional

import numpy as np
from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver
from src.methods.utils import create_population_file, load_memmap


class PopulationHandler:
    """Manages the creation, access, and safe closure of the population's memmap file.

    The memmap file holds the entire genome matrix and allows for efficient
    disk-based data handling during the genetic algorithm runtime.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        paths: PathResolver,
        genome_length: int,
        filename_constant: str,
        weight_sum: int,
    ) -> None:
        """Initializes the handler, creates initial population, and loads memmap.

        The memmap handle is loaded in read-only mode after file creation.

        Args:
            config (ExperimentConfig): Configuration, including stream size
                                       and RNG.
            paths (PathResolver): Resolver for accessing the temporary directory.
            genome_length (int): The number of genes in an individual's genome.
            filename_constant (str): Unique identifier for the experiment files.
            weight_sum (int): Total weight sum of all items, used for probability
                              calculation.

        Raises:
            ValueError: If required config fields (stream_batch_size or rng)
                        are None.
        """
        self.population_size = config.population_size
        self.genome_length = genome_length
        if config.stream_batch_size is None or config.rng is None:
            raise ValueError("Config is corrupted!")
        self.stream_batch = config.stream_batch_size
        self.rng = config.rng
        self.q = config.generate_probability_of_failure(weight_sum)
        self.filename_constant = filename_constant
        self.temp_path = paths.get_temp_path()

        create_population_file(
            temp=self.temp_path,
            population_size=self.population_size,
            genome_length=self.genome_length,
            stream_batch=self.stream_batch,
            rng=self.rng,
            probability_of_failure=self.q,
            filename_constant=self.filename_constant,
        )
        self.pop_handle: Optional[np.memmap[tuple[int, int], np.dtype[np.uint8]]]
        self.pop_handle, self.pop_config = load_memmap(
            filename_constant=self.filename_constant,
            open_mode="r",
            temp=self.temp_path,
        )

    def get_pop_handle(
        self,
    ) -> Optional[np.memmap[tuple[int, int], np.dtype[np.uint8]]]:
        """Returns the current memmap handle for the population file.

        Returns:
            Optional[np.memmap]: The active memmap handle, or None if closed.
        """
        return self.pop_handle

    def get_pop_config(self) -> dict:
        """Returns the configuration dictionary loaded from the population JSON file.

        Returns:
            dict: Dictionary containing memmap metadata (filesize, dtype, shape, etc.).
        """
        return self.pop_config

    def open_pop(
        self,
        open_mode: Literal[
            "readonly", "r", "copyonwrite", "c", "readwrite", "r+", "write", "w+"
        ] = "r",
    ) -> None:
        """Opens or re-opens the population memmap file with the specified access mode.

        This allows switching between read-only and writeable modes as needed.

        Args:
            open_mode (Literal): The mode to open the memmap file (e.g., 'r', 'r+').
                                 Defaults to 'r' (read-only).
        """
        if self.pop_handle is None:
            self.pop_handle, _ = load_memmap(
                filename_constant=self.filename_constant,
                open_mode=open_mode,
                temp=self.temp_path,
            )

    def close(self) -> None:
        """Safely closes the memmap resource.

        Flushes data to disk, deletes the handle, and forces garbage collection
        to ensure system resources are released promptly.
        """
        if self.pop_handle is not None:
            h = self.pop_handle
            self.pop_handle = None
            h.flush()
            del h
            import gc

            gc.collect()
