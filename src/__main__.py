import sys
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
MAX_WEIGHT = 2000
CROSSOVER_PROBABILITY = 1
MUTATION_PROBABILITY = 0.1
ITERATIONS = 500
POPULATION_SIZE = int(1e5)

ITEMS_VALUE_WEIGHT = methods.utils.load_data(LOW_DIMENSIONAL_PATH / "f3_l-d_kp_4_20")
methods.utils.create_population_file(
    population_size=POPULATION_SIZE,
    genome_length=len(ITEMS_VALUE_WEIGHT),
    stream_batch=500,
    SEED=SEED,
)
fitness = methods.fitness_score.calc_fitness_score(ITEMS_VALUE_WEIGHT, MAX_WEIGHT)
for i in range(ITERATIONS):
    parent_pool = methods.selection_methods.fitness_proportionate_selection(
        fitness, parent_group_size=POPULATION_SIZE, seed=SEED
    )
    parent_pairs = parent_pairing(parent_pool)
    single_crossover(
        parent_pairs,
        seed=SEED,
        cross_propab=CROSSOVER_PROBABILITY,
        mutation_probab=MUTATION_PROBABILITY,
    )
    fitness = methods.fitness_score.calc_fitness_score(ITEMS_VALUE_WEIGHT, MAX_WEIGHT)
    print(f"iteration {i}: {fitness}")


files = ["population.dat", "population.json"]
for file in files:
    if os.path.exists(file):
        os.remove(file)
        print(f"removed {file}")
