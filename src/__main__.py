import numpy as np
from pathlib import Path
from src.methods.utils import (
    load_data,
    create_population_file,
    clear_temp_files,
    log_output,
    final_screen,
    load_memmap,
)
from src.methods.fitness_score import calc_fitness_score
from src.methods.selection_methods import fitness_proportionate_selection
from src.methods.reproduction_tools import (
    single_crossover,
    parent_pairing,
    single_crossover_batched,
)
from src.methods.experiment_defining_tools import create_unique_experiment_name

#############################################################################
PROJECT_PATH = Path(__file__).resolve().parent.parent
LARGE_SCALE_PATH = PROJECT_PATH / "dane AG 2" / "large_scale"
LOW_SCALE_PATH = PROJECT_PATH / "dane AG 2" / "low-dimensional"

LOW_SCALE_OPTIMUM_PATH = PROJECT_PATH / "dane AG 2" / "low-dimensional-optimum"
LARGE_SCALE_OPTIMUM_PATH = PROJECT_PATH / "dane AG 2" / "large_scale-optimum"
FILENAME = "knapPI_2_100_1000_1"

SEED = None
MAX_WEIGHT = 1000
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.001
ITERATIONS = int(100)
POPULATION_SIZE = int(1e5)
PENALTY_PERCENTAGE = 1
EXPERIMENT_NO = 1
#################################################################################

try:
    if SEED is not None:
        rng = np.random.default_rng(seed=SEED)
    else:
        rng = np.random.default_rng()

    value_weight_dict = load_data(path=LARGE_SCALE_PATH / FILENAME)

    probability_of_failure = MAX_WEIGHT / sum(
        value_weight_dict[i][1] for i in range(len(value_weight_dict))
    )
    # define file naming constant for specific experiment
    experiment_name = create_unique_experiment_name(
        filename=FILENAME,
        population_length=POPULATION_SIZE,
        genome_width=len(value_weight_dict),
        number_of_generations=ITERATIONS,
        crossover=CROSSOVER_PROBABILITY,
        mutation=MUTATION_PROBABILITY,
        exp_no=EXPERIMENT_NO,
    )

    create_population_file(
        filename_constant=experiment_name,
        population_size=POPULATION_SIZE,
        genome_length=len(value_weight_dict),
        stream_batch=500,
        rng=rng,
        probability_of_failure=probability_of_failure,
    )

    log_output(
        filename_constant=experiment_name,
        message=str(
            "Population initialized with probability of failure threshold 'q' ="
            + str(probability_of_failure)
        ),
    )

    population_file_handle, population_file_config = load_memmap(
        filename_constant=experiment_name
    )
    fitness = calc_fitness_score(
        value_weight_dict=value_weight_dict,
        max_weight=MAX_WEIGHT,
        penalty=PENALTY_PERCENTAGE,
        population_file_handle=population_file_handle,
        population_file_config=population_file_config,
    )
    best_idx = fitness[:, 0].argmax()
    best_score, weight = fitness[best_idx]
    log_output(
        filename_constant=experiment_name,
        iteration=0,
        best_genome_index=best_idx,
        fitness=best_score,
        weight=weight,
        genome=population_file_handle[best_idx],
        message=str("Population created successfully as iteration 0"),
    )
except Exception as e:
    log_output(message=str(e))
for i in range(1, ITERATIONS + 1):
    population_file_handle, population_file_config = load_memmap(
        filename_constant=experiment_name
    )
    parent_pool = fitness_proportionate_selection(
        fitness_score=fitness,
        parent_group_size=POPULATION_SIZE,
        rng=rng,
        population_file_config=population_file_config,
    )
    parent_pairs = parent_pairing(parent_pool=parent_pool, rng=rng)
    single_crossover_batched(
        parent_pairs=parent_pairs,
        rng=rng,
        crossover_probability=CROSSOVER_PROBABILITY,
        mutation_probability=MUTATION_PROBABILITY,
        population_file_handle=population_file_handle,
        population_file_config=population_file_config,
    )
    fitness = calc_fitness_score(
        value_weight_dict=value_weight_dict,
        max_weight=MAX_WEIGHT,
        penalty=PENALTY_PERCENTAGE,
        population_file_handle=population_file_handle,
        population_file_config=population_file_config,
    )

    best_idx = fitness[:, 0].argmax()
    best_score, weight = fitness[best_idx]
    log_output(
        filename_constant=experiment_name,
        iteration=i,
        best_genome_index=best_idx,
        fitness=best_score,
        weight=weight,
        genome=population_file_handle[best_idx],
    )
    print(f"Finished calculating iteration {i}")

final_screen()
clear_temp_files()
