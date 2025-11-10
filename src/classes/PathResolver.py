import os
import shutil
from pathlib import Path
from shutil import rmtree


class PathResolver:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    def __init__(self):
        self.filename_constant: str | None = None
        self.temp_dir: Path | None = None
        self.output_dir: Path | None = None
        self.logging_dir: Path | None = None

        self.small_scale_path = (
            Path(self.PROJECT_ROOT) / "dane AG 2" / "low-dimensional"
        )
        self.large_scale_path = Path(self.PROJECT_ROOT) / "dane AG 2" / "large_scale"

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
        self.filename_constant = filename_constant
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logging_dir.mkdir(parents=True, exist_ok=True)

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

    def clean_children_temp(self):
        if Path(self.temp_dir / f"child_{self.filename_constant}.dat").exists():
            os.remove(self.temp_dir / f"{self.filename_constant}.dat")
            os.rename(
                src=Path(self.temp_dir / f"child_{self.filename_constant}.dat"),
                dst=Path(self.temp_dir / f"{self.filename_constant}.dat"),
            )
