# --> IMPORTS <--
import numpy as np

from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PopulationHandler import PopulationHandler


def calc_fitness_score_batched(
        value_weight_arr: np.ndarray,
        config: ExperimentConfig,
        pop_manager: PopulationHandler,
):

    max_weight = config.max_weight
    penalty_factor = config.penalty

    population = pop_manager.get_pop_handle()
    batch = config.stream_batch_size
    value = value_weight_arr[:, 0]
    weight = value_weight_arr[:, 1]

    fitness_score = np.zeros(shape=(population.shape[0],2), dtype=np.int64)

    for start in range(0, population.shape[0], batch):
        stop = min(start + batch, population.shape[0])
        current_batch = population[start:stop]
        calculated_scores = current_batch @ value
        calculated_weights = current_batch @ weight
        over_limit_mask = calculated_weights > max_weight
        if penalty_factor == 1:
            penalty_value = calculated_scores
        else:
            penalty_value = np.maximum(0, calculated_weights-max_weight)*penalty_factor
        penalized_score = np.where(over_limit_mask, calculated_scores-penalty_value,calculated_scores)

        fitness_score[start:stop] = np.array((penalized_score, calculated_weights)).transpose()
    return fitness_score




