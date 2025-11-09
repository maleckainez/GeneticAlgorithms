import shutil
from pathlib import Path


class PathResolver:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    def __init__(self):
        self.temp_dir = Path(self.PROJECT_ROOT) / "temp"
        self.output_dir = Path(self.PROJECT_ROOT) / "output"

        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.small_scale_path = (
            Path(self.PROJECT_ROOT) / "dane AG 2" / "low-dimensional"
        )
        self.large_scale_path = Path(self.PROJECT_ROOT) / "dane AG 2" / "large_scale"

        self.data_path: Path | None = None

    def get_temp_path(self) -> Path:
        return self.temp_dir

    def get_output_dir(self) -> Path:
        return self.output_dir

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
