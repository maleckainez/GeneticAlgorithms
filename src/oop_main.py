from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver
from src.methods.utils import load_yaml_config, load_data

input_config = load_yaml_config(
    "config.yaml"
)  # Loads yaml file and flatten it to config dict
config = ExperimentConfig(
    **input_config
)  # Creates class instance handling config values
paths = PathResolver()  # Creates class instance handling filepaths
value_weight_dict = load_data(
    paths.get_dict_filepath(config.data_filename)
)  # Gets dictionary of values and weights
