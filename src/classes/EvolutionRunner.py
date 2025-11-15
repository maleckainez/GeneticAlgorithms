import numpy as np

from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver
from src.classes.PopulationHandler import PopulationHandler as PopHandler
from src.classes.Reproduction import Reproduction
from src.methods.experiment_defining_tools import create_unique_experiment_name
from src.methods.fitness_score import calc_fitness_score_batched
from src.methods.utils import load_data
import src.methods.logging_library as log
from src.methods.selection_methods import (
    roulette_selection,
    tournament_selection,
    linear_rank_selection,
)
from src.classes.ChildrenHandler import ChildrenHandler

SELECTION_METHODS = {
    "roulette": roulette_selection,
    "tournament": tournament_selection,
    "rank": linear_rank_selection,
}
CROSSOVER_METHODS = {
    "one": Reproduction.single_crossover,
    "two": Reproduction.double_crossover,
}


class EvolutionRunner:
    def __init__(self, input_config):

        self._load_configuration(input_config)

        self._prepare_environment()

        self._initialize_first_generation()

        self._load_strategies()

    def _load_configuration(self,input_config):

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
        best_idx, best_score, weight, repetitions = self._get_best_individual()
        self.logger.info(f"Population created successfully as iteration 0")
        log.generation(
            logger=self.logger,
            best_idx=best_idx,
            best_score=best_score,
            weight=weight,
            iteration=0,
            repetitions=repetitions
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
        for iteration in range(1, self.generations+1):
            parent_pool = self.selection_function(
                fitness_arr=self.fitness, config=self.config
            )
            children_manager = ChildrenHandler(config=self.config, paths=self.paths, genome_length=self.population_manager.genome_length)
            crossover = Reproduction(parent_pool, self.config, self.paths)
            method_name = self.crossover_function.__name__
            getattr(crossover, method_name)(self.population_manager, children_manager)
            self._clean_children(children_manager)
            self.population_manager.open_pop()
            self.fitness = calc_fitness_score_batched(
                value_weight_arr=self.value_weight_array,
                config=self.config,
                pop_manager=self.population_manager,
            )
            best_idx, best_score, weight, repetitions = self._get_best_individual()
            log.generation(
                logger=self.logger,
                best_idx=best_idx,
                best_score=best_score,
                weight=weight,
                iteration=iteration,
                repetitions=repetitions,
            )
    def _get_best_individual(self):
        sorted_fitness_descending = np.lexsort((self.fitness[:,1], -self.fitness[:,0]))
        best_idx = sorted_fitness_descending[0]
        best_score, weight = self.fitness[best_idx]
        mask = (self.fitness[:,0] == best_score) & (self.fitness[:,1] == weight)
        number_of_identical_best = np.sum(mask)
        return best_idx, best_score, weight, number_of_identical_best

    def _clean_children(self, children_manager: ChildrenHandler):
        children_manager.close()
        self.population_manager.close()
        pop_config = self.population_manager.get_pop_config()
        filesize = pop_config["filesize"]
        self.paths.commit_children(expected_size=filesize)
