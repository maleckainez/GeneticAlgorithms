from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver
from src.methods.utils import create_population_file, load_memmap


class PopulationHandler:

    def __init__(
        self,
        config: ExperimentConfig,
        paths: PathResolver,
        genome_length: int,
        filename_constant: str,
        weight_sum: int,
    ) -> None:
        self.population_size = config.population_size
        self.genome_length = genome_length
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

        self.pop_handle, self.pop_config = load_memmap(
            filename_constant=self.filename_constant,
            open_mode="r",
            temp=self.temp_path,
        )

    def get_pop_handle(self):
        return self.pop_handle

    def get_pop_config(self):
        return self.pop_config

    def open_pop(self, open_mode: str = "r") -> None:
        if self.pop_handle is None:
            self.pop_handle = load_memmap(
                filename_constant=self.filename_constant,
                open_mode=open_mode,
                temp=self.temp_path,
            )

    def close(self):
        if self.pop_handle is not None:
            self.pop_handle.flush()
            del self.pop_handle
            self.pop_handle = None
