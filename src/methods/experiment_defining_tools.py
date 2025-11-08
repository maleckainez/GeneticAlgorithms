import re

def create_unique_experiment_name(
        filename: str,
        population_length:int,
        genome_width:int,
        number_of_generations: int,
        crossover: float,
        mutation: float,
        exp_no:int
) -> str:
    cr = re.sub(r'\.','p',f"{crossover}")
    mr = re.sub(r'\.','p',f"{mutation}")
    fname = re.sub("[^A-Za-z0-9]+", "", filename)
    parts = [
        fname,
        f"PS{population_length}",
        f"GW{genome_width}",
        f"GE{number_of_generations}",
        f"CR{cr}",
        f"MR{mr}",
        f"EXP{exp_no:03d}",
    ]
    unique_id = '-'.join(parts)
    return unique_id
