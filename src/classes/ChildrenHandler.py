import numpy as np
from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver


class ChildrenHandler:
    def __init__(
        self,
        config: ExperimentConfig,
        paths: PathResolver,
        genome_length: int,
    ) -> None:
        self.population_size = config.population_size
        self.genome_length = genome_length
        self.stream_batch = config.stream_batch_size
        self.temp_path = paths.get_temp_path()
        self.children_handle = self._create_mmap(paths)

    def _create_mmap(self, paths):
        return np.memmap(
            filename=paths.get_temp_path() / f"child_{paths.filename_constant}.dat",
            shape=(self.population_size, self.genome_length),
            dtype=np.uint8,
            mode="w+",
        )

    def get_children_handle(self):
        return self.children_handle

    def close(self):
        if self.children_handle is not None:
            self.children_handle.flush()
            del self.children_handle
            self.children_handle = None
