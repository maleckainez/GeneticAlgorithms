"""Generate unique, timestamped experiment identifiers based on configuration."""

import datetime
import re

from src.classes.ExperimentConfig import ExperimentConfig


def name_generator(
    config: ExperimentConfig,
    genome_length: int,
) -> str:
    """Generate a unique experiment name based on configuration and genome length.

    Args:
        config (ExperimentConfig): Experiment configuration object.
        genome_length (int): Length of the genome used in the experiment.

    Returns:
        str: Generated experiment identifier string.
    """
    method = config.selection_type
    crossover = config.crossover_probability
    mutation = config.mutation_probability
    filename = config.data_filename
    population_length = config.population_size
    genome_length = genome_length
    number_of_generations = config.generations
    exp_no = config.experiment_identifier

    cr = re.sub(r"\.", "p", f"{crossover}")
    mr = re.sub(r"\.", "p", f"{mutation}")
    fname = re.sub("[^A-Za-z0-9]+", "_", filename)
    parts = [
        fname,
        method,
        f"PS{population_length}",
        f"GW{genome_length}",
        f"GE{number_of_generations}",
        f"CR{cr}",
        f"MR{mr}",
        f"EXP{exp_no:03d}" f"T{datetime.datetime.now().strftime('%M%S')}",
    ]
    unique_id = "-".join(parts)
    return unique_id
