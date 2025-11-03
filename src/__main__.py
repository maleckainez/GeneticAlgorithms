import os
import numpy as np
from pathlib import Path
from src.methods.utils import load_data, create_population_file
from src.methods.fitness_score import calc_fitness_score
from src.methods.selection_methods import fitness_proportionate_selection
from src.methods.reproduction_tools import single_crossover, parent_pairing


#############################################################################
PROJECT_PATH = Path(__file__).resolve().parent.parent
LARGE_SCALE_PATH = PROJECT_PATH / "dane AG 2" / "large_scale"
LOW_SCALE_PATH = PROJECT_PATH / "dane AG 2" / "low-dimensional"

LOW_SCALE_OPTIMUM_PATH = PROJECT_PATH / "dane AG 2" / "low-dimensional-optimum"
LARGE_SCALE_OPTIMUM_PATH = PROJECT_PATH / "dane AG 2" / "large_scale-optimum"
FILENAME = "knapPI_2_100_1000_1"

SEED = 1111
MAX_WEIGHT = 1000
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.1
ITERATIONS = int(1e6)
POPULATION_SIZE = int(1e2)
PENALTY_PERCENTAGE = 1
#################################################################################

if SEED is not None:
    rng = np.random.default_rng(seed=SEED)
else:
    rng = np.random.default_rng()

value_weight_dict = load_data(path=LARGE_SCALE_PATH / FILENAME)

propab_q = MAX_WEIGHT / sum(
    value_weight_dict[i][1] for i in range(len(value_weight_dict))
)

create_population_file(
    population_size=POPULATION_SIZE,
    genome_length=len(value_weight_dict),
    stream_batch=500,
    rng=rng,
    q=propab_q,
)

fitness = calc_fitness_score(
    value_weight_dict=value_weight_dict,
    max_weight=MAX_WEIGHT,
    penalty=PENALTY_PERCENTAGE,
)

for i in range(ITERATIONS):
    parent_pool = fitness_proportionate_selection(
        fitness_score=fitness, parent_group_size=POPULATION_SIZE, rng=rng
    )
    parent_pairs = parent_pairing(parent_pool=parent_pool, rng=rng)
    single_crossover(
        parent_pairs=parent_pairs,
        rng=rng,
        cross_propab=CROSSOVER_PROBABILITY,
        mutation_probab=MUTATION_PROBABILITY,
    )
    fitness = calc_fitness_score(
        value_weight_dict=value_weight_dict,
        max_weight=MAX_WEIGHT,
        penalty=PENALTY_PERCENTAGE,
    )
    print(f"iteration {i}")

files = ["population.dat", "population.json"]
for file in files:
    if os.path.exists(file):
        os.remove(file)
        print(f"removed {file}")
