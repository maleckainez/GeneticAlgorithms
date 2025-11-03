import sys

import numpy as np

import methods.utils
import methods.fitness_score
import methods.selection_methods
import methods.reproduction_tools
import os
from pathlib import Path
from src.methods.reproduction_tools import single_crossover, parent_pairing


#############################################################################
PROJECT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_PATH / "src"))
LARGE_SCALE_PATH = PROJECT_PATH / "dane AG 2" / "large_scale"
LOW_DIMENSIONAL_PATH = PROJECT_PATH / "dane AG 2" / "low-dimensional"
SEED = 2137
MAX_WEIGHT = 20
CROSSOVER_PROBABILITY = 1
MUTATION_PROBABILITY = 0.1
ITERATIONS = 100
POPULATION_SIZE = 20

rng = np.random.default_rng(seed=SEED)
ITEMS_VALUE_WEIGHT = methods.utils.load_data(LOW_DIMENSIONAL_PATH / "f3_l-d_kp_4_20")
methods.utils.create_population_file(
    population_size=POPULATION_SIZE,
    genome_length=len(ITEMS_VALUE_WEIGHT),
    stream_batch=500,
    rng=rng,
)
fitness = methods.fitness_score.calc_fitness_score(ITEMS_VALUE_WEIGHT, MAX_WEIGHT)
for i in range(ITERATIONS):
    parent_pool = methods.selection_methods.fitness_proportionate_selection(
        fitness, parent_group_size=POPULATION_SIZE, rng=rng
    )
    parent_pairs = parent_pairing(parent_pool)
    single_crossover(
        parent_pairs,
        rng=rng,
        cross_propab=CROSSOVER_PROBABILITY,
        mutation_probab=MUTATION_PROBABILITY,
    )
    fitness = methods.fitness_score.calc_fitness_score(ITEMS_VALUE_WEIGHT, MAX_WEIGHT)
    best_idx = fitness[:, 0].argmax()
    best_score, best_weight = fitness[best_idx]
    print(
        f"iter={i:03d} | best={best_score} (idx {best_idx}) | "
        f"weight={best_weight} | avg={fitness[:, 0].mean():.1f}"
    )


files = ["population.dat", "population.json"]
for file in files:
    if os.path.exists(file):
        os.remove(file)
        print(f"removed {file}")
