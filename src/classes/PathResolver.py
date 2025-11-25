"""Module for resolving and managing all file paths within the project structure."""

import os
import shutil
import time
from pathlib import Path


class PathResolver:
    """Manages creation, resolution, and cleanup of all experiment-specific paths."""

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    def __init__(self) -> None:
        """Initializes PathResolver with default path constants.

        All dynamic directories (temp, output, logs, plots) are initially set to
        None and must be initialized by calling the `initialize()` method.
        """
        self.filename_constant: str = "undefined_experiment"
        self.temp_dir: Path | None = None
        self.output_dir: Path | None = None
        self.logging_dir: Path | None = None
        self.plot_dir: Path | None = None

        self.small_scale_path = (
            Path(self.PROJECT_ROOT) / "dane AG 2" / "low-dimensional"
        )
        self.small_scale_optimum = (
            Path(self.PROJECT_ROOT) / "dane AG 2" / "low-dimensional-optimum"
        )
        self.large_scale_path = Path(self.PROJECT_ROOT) / "dane AG 2" / "large_scale"
        self.large_scale_optimum = (
            Path(self.PROJECT_ROOT) / "dane AG 2" / "large_scale-optimum"
        )

        self.data_path: Path | None = None

    def initialize(self, filename_constant: str) -> None:
        """Sets the experiment identifier & creates the run_output directory structure.

        Args:
            filename_constant (str): Unique identifier for the current experiment run.
        """
        self.temp_dir = (
            Path(self.PROJECT_ROOT) / "run_output" / f"{filename_constant}" / "temp"
        )
        self.output_dir = (
            Path(self.PROJECT_ROOT) / "run_output" / f"{filename_constant}" / "output"
        )
        self.logging_dir = (
            Path(self.PROJECT_ROOT) / "run_output" / f"{filename_constant}" / "logs"
        )
        self.plot_dir = (
            Path(self.PROJECT_ROOT) / "run_output" / f"{filename_constant}" / "plots"
        )
        self.filename_constant = filename_constant
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def get_temp_path(self) -> Path:
        """Returns the absolute path to the temporary files directory.

        Raises:
            RuntimeError: If the directories were not initialized via `initialize()`.

        Returns:
            Path: The path to the temp directory.
        """
        if self.temp_dir is None:
            raise RuntimeError("Directories were not initialized")
        return self.temp_dir

    def get_output_path(self) -> Path:
        """Returns the absolute path to the final output directory.

        Raises:
            RuntimeError: If the directories were not initialized via `initialize()`.

        Returns:
            Path: The path to the output directory.
        """
        if self.output_dir is None:
            raise RuntimeError("Directories were not initialized")
        return self.output_dir

    def get_logging_path(self) -> Path:
        """Returns the absolute path to the logging directory.

        Raises:
            RuntimeError: If the directories were not initialized via `initialize()`.

        Returns:
            Path: The path to the logging directory.
        """
        if self.logging_dir is None:
            raise RuntimeError("Directories were not initialized")
        return self.logging_dir

    def cleanup_temp_dir(self) -> None:
        """Safely removes the temporary directory and all its contents."""
        if isinstance(self.temp_dir, Path) and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def get_dict_filepath(self, file_name: str) -> Path:
        """Resolves the absolute path to the input data file based on its name prefix.

        The path is stored internally as `self.data_path`.

        Args:
            file_name (str): The name of the data file
                             (e.g., 'knapPI_1_100_1000_1' or 'f6_l-d_kp_10_60').

        Raises:
            Exception: If the file name prefix is invalid (not 'knap' or 'f').

        Returns:
            Path: The absolute path to the data file.
        """
        if file_name.startswith("knap"):
            path = self.large_scale_path / file_name
        elif file_name.startswith("f"):
            path = self.small_scale_path / file_name
        else:
            raise Exception("Invalid file name")
        self.data_path = path
        return self.data_path

    def get_children_filepath(self) -> Path:
        """Returns the absolute path for the temporary children memmap file.

        Returns:
            Path: Path to the children data file
                  (e.g., .../temp/child_experimentID.dat).
        """
        temp_directory = self.get_temp_path()
        return temp_directory / f"child_{self.filename_constant}.dat"

    def commit_children(self, expected_size: int, retries: int = 3) -> None:
        """Replaces the old population memmap file with the new children file.

        Uses os.replace() to minimize the risk of data corruption. Retries are
        implemented to handle temporary PermissionErrors.

        Args:
            expected_size (int): The required file size of the children file
                                 for validation.
            retries (int, optional): Number of attempts to commit the file
                                     replacement. Defaults to 3.

        Raises:
            RuntimeError: If the children file is missing, size mismatch occurs,
                          or the commit fails after all retries.
        """
        temp_directory = self.get_temp_path()
        child = self.get_children_filepath()
        population = Path(temp_directory / f"{self.filename_constant}.dat")
        if not child.exists():
            raise RuntimeError(f"Missing children file {child}")
        if child.stat().st_size != expected_size:
            raise RuntimeError(
                f"Children size mismatch: {child.stat().st_size} =/= {expected_size}"
            )
        last_error = None
        for _ in range(retries):
            try:
                os.replace(child, population)
                return
            except PermissionError as err:
                last_error = err
                time.sleep(0.2)
        raise RuntimeError(
            f"Commit failed after {retries} tries.\n"
            f"Dst: {population}\n"
            f"Src: {child}\n"
            f"With error: {last_error}"
        )

    def get_plot_path(self) -> Path:
        """Returns the absolute path to the plots output directory.

        Raises:
            RuntimeError: If the directories were not initialized via `initialize()`.

        Returns:
            Path: The path to the plots directory.
        """
        if self.plot_dir is None:
            raise RuntimeError("Directories were not initialized")
        return self.plot_dir

    def get_optimum_path(self) -> Path:
        """Returns the base directory path for the optimum solution files.

        The path is determined by the prefix of the experiment's filename constant.

        Raises:
            Exception: If the filename constant prefix is invalid.

        Returns:
            Path: The path to the directory containing optimum solutions.
        """
        if self.filename_constant.startswith("knap"):
            return self.large_scale_optimum
        elif self.filename_constant.startswith("f"):
            return self.small_scale_optimum
        else:
            raise Exception("Invalid file name")
