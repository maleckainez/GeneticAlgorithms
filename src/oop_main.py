from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver
from src.classes.PopulationHandler import PopulationHandler as PopHandler
from src.methods.experiment_defining_tools import create_unique_experiment_name
from src.methods.fitness_score import calc_fitness_score_batched
from src.methods.reproduction_tools import (
    single_crossover_batched,
    double_crossover_batched,
    parent_pairing,
)
from src.methods.utils import load_yaml_config, load_data
import src.methods.logging_library as log
from src.methods.selection_methods import (
    roulette_selection,
    tournament_selection,
    linear_rank_selection,
)

SELECTION_METHODS = {
    "roulette": roulette_selection,
    "tournament": tournament_selection,
    "rank": linear_rank_selection,
}
CROSSOVER_METHODS = {
    "one": single_crossover_batched,
    "two": double_crossover_batched,
}


class EvolutionRunner:
    def __init__(self):

        self._load_configuration()

        self._prepare_environment()

        self._initialize_first_generation()

        self._load_strategies()

    def _load_configuration(self):
        # Loads yaml file and flatten it to config dict
        input_config = load_yaml_config("config.yaml")

        # Creates class instance handling config values
        self.config = ExperimentConfig(**input_config)

        # Creates class instance handling filepaths
        self.paths = PathResolver()

        # Gets dictionary of values and weights
        self.value_weight_array = load_data(
            self.paths.get_dict_filepath(self.config.data_filename)
        )

        self.generations = self.config.generations

    def _prepare_environment(self):
        filename_constant = create_unique_experiment_name(
            config=self.config,
            genome_length=self.value_weight_array.shape[0],
        )

        self.paths.initialize(filename_constant=filename_constant)

        self.logger = log.initialize(config=self.config, paths=self.paths)

    def _initialize_first_generation(self):
        self.population_manager = PopHandler(
            config=self.config,
            paths=self.paths,
            genome_length=self.value_weight_array.shape[0],
            filename_constant=self.paths.filename_constant,
            weight_sum=self.value_weight_array[:, 1].sum(),
        )
        self.fitness = calc_fitness_score_batched(
            value_weight_arr=self.value_weight_array,
            config=self.config,
            pop_manager=self.population_manager,
        )
        best_idx = self.fitness[:, 0].argmax()
        best_score, weight = self.fitness[best_idx]
        self.logger.info(f"Population created successfully as iteration 0")
        log.generation(
            logger=self.logger,
            best_idx=best_idx,
            best_score=best_score,
            weight=weight,
            iteration=0,
        )

    def _load_strategies(self):
        selection_type = self.config.selection_type
        crossover_type = self.config.crossover_type
        if selection_type not in SELECTION_METHODS:
            self.logger.critical(f"Invalid selection method: {selection_type}")
            raise ValueError(f"Invalid selection method: {selection_type}")
        self.selection_function = SELECTION_METHODS[selection_type]
        self.logger.info(f"{selection_type} selection method was chosen.")
        if crossover_type not in CROSSOVER_METHODS:
            self.logger.critical(f"Invalid crossover method: {crossover_type}")
            raise ValueError(f"Invalid crossover method: {crossover_type}")
        self.crossover_function = CROSSOVER_METHODS[crossover_type]
        self.logger.info(f"{crossover_type} crossover method was chosen.")

    def evolve(self):
        for iteration in range(self.generations):
            parent_pool = self.selection_function(
                fitness_arr=self.fitness, config=self.config
            )
            parent_pairs = parent_pairing(parent_pool=parent_pool, config=self.config)
            self.crossover_function(
                parent_pairs=parent_pairs,
                config=self.config,
                paths=self.paths,
                pop_manager=self.population_manager,
            )
            self._clean_children()
            self.population_manager.open_pop()
            fitness = calc_fitness_score_batched(
                value_weight_arr=self.value_weight_array,
                config=self.config,
                pop_manager=self.population_manager,
            )
            best_idx = fitness[:, 0].argmax()
            best_score, weight = fitness[best_idx]
            log.generation(
                logger=self.logger,
                best_idx=best_idx,
                best_score=best_score,
                weight=weight,
                iteration=iteration,
            )

    def _clean_children(self):
        self.population_manager.close()
        self.paths.clean_children_temp()


if __name__ == "__main__":
    EvolutionRunner().evolve()
