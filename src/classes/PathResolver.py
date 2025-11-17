import os
import shutil
from pathlib import Path
from shutil import rmtree
import time


class PathResolver:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    def __init__(self):
        self.filename_constant: str | None = None
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
        if self.temp_dir is None:
            raise RuntimeError("Directories were not initialized")
        return self.temp_dir

    def get_output_path(self) -> Path:
        if self.output_dir is None:
            raise RuntimeError("Directories were not initialized")
        return self.output_dir

    def get_logging_path(self) -> Path:
        if self.logging_dir is None:
            raise RuntimeError("Directories were not initialized")
        return self.logging_dir

    def cleanup_temp_dir(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def get_dict_filepath(self, file_name: str) -> Path:
        if file_name.startswith("knap"):
            path = self.large_scale_path / file_name
        elif file_name.startswith("f"):
            path = self.small_scale_path / file_name
        else:
            raise Exception("Invalid file name")
        self.data_path = path
        return self.data_path

    def get_children_filepath(self):
        return self.temp_dir / f"child_{self.filename_constant}.dat"

    def commit_children(self, expected_size: int, retries: int = 10):
        child = self.get_children_filepath()
        population = Path(self.temp_dir / f"{self.filename_constant}.dat")
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
            f"Commit failed after {retries} tries.\nDst: {population}\nSrc: {child}\nWith error: {last_error}"
        )

    def get_plot_path(self):
        if self.plot_dir is None:
            raise RuntimeError("Directories were not initialized")
        return self.plot_dir

    def get_optimum_path(self):
        if self.filename_constant.startswith("knap"):
            return self.large_scale_optimum
        elif self.filename_constant.startswith("f"):
            return self.small_scale_optimum
        else:
            raise Exception("Invalid file name")
