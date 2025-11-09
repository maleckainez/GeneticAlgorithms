from src.classes.ExperimentConfig import ExperimentConfig
from src.methods.utils import create_population_file, load_memmap


class PopulationHandler:

    def __init__(
        self,
        config: ExperimentConfig,
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

        create_population_file(
            population_size=self.population_size,
            genome_length=self.genome_length,
            stream_batch=self.stream_batch,
            rng=self.rng,
            probability_of_failure=self.q,
            filename_constant=self.filename_constant,
        )

        self.pop_handle, self.pop_config = load_memmap(
            filename_constant=self.filename_constant, open_mode="r"
        )

    def get_population_handle(self):
        return self.pop_handle

    def get_population_config(self):
        return self.pop_config
