"""Module for creation and management of memory-mapped children file."""

from typing import Optional

import numpy as np
from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver


class ChildrenHandler:
    """Handles the creation, management, and safe closure of the children memmap."""

    def __init__(
        self,
        config: ExperimentConfig,
        paths: PathResolver,
        genome_length: int,
    ) -> None:
        """Creates children memmap.

        Args:
            config (ExperimentConfig): Configuration object containing
                ``population size`` and ``batch size`` variables.
            paths (PathResolver): Utility class for generating temporary file paths.
            genome_length (int): The number of genes (columns) in each
                individual's genome (line).
        """
        self.population_size = config.population_size
        self.genome_length = genome_length
        self.stream_batch = config.stream_batch_size
        self.temp_path = paths.get_temp_path()
        self.children_handle: Optional[np.memmap[tuple[int, int], np.dtype[np.uint8]]]
        self.children_handle = self._create_mmap(paths)

    def _create_mmap(
        self, paths: PathResolver
    ) -> np.memmap[tuple[int, int], np.dtype[np.uint8]]:
        """Creates and initializes the memory-mapped file on disk.

        Args:
            paths (PathResolver): Utility class used to construct the full file path.

        Returns:
            np.memmap: A writeable memory-mapped array of shape
                ``(population_size, genome_length)``.
        """
        return np.memmap(
            filename=paths.get_temp_path() / f"child_{paths.filename_constant}.dat",
            shape=(self.population_size, self.genome_length),
            dtype=np.uint8,
            mode="w+",
        )

    def get_children_handle(
        self,
    ) -> Optional[np.memmap[tuple[int, int], np.dtype[np.uint8]]]:
        """Returns the current memory-map handle for direct reading or writing.

        Returns:
            Optional[np.memmap]: The active memmap handle, or ``None``
                if the resource has been closed.
        """
        return self.children_handle

    def close(self) -> None:
        """Closes the memory-mapped file resource and releases system resources.

        This method flushes data to disk, deletes the memmap handle, and calls
        the garbage collector to ensure prompt cleanup.
        """
        if self.children_handle is not None:
            h = self.children_handle
            self.children_handle = None
            h.flush()
            del h
            import gc

            gc.collect()
