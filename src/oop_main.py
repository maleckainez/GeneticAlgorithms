from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver
from src.classes.PopulationHandler import PopulationHandler as PopHandler
from src.methods.experiment_defining_tools import create_unique_experiment_name
from src.methods.fitness_score import calc_fitness_score, calc_fitness_score_batched
from src.methods.utils import load_yaml_config, load_data

# Loads yaml file and flatten it to config dict
input_config = load_yaml_config("config.yaml")

# Creates class instance handling config values
config = ExperimentConfig(**input_config)

# Creates class instance handling filepaths
paths = PathResolver()

# Gets dictionary of values and weights
value_weight_array = load_data(paths.get_dict_filepath(config.data_filename))

# Creates unique filename constant
filename_constant = create_unique_experiment_name(
    config=config,
    genome_length=value_weight_array.shape[0],
)

# Creates starter population mmap
population_manager = PopHandler(
    config=config,
    paths=paths,
    genome_length= value_weight_array.shape[0],
    filename_constant=filename_constant,
    weight_sum=value_weight_array[:, 1].sum()
)

fitness = calc_fitness_score_batched(
    value_weight_arr=value_weight_array,
    config=config,
    pop_manager=population_manager
)

print(fitness)