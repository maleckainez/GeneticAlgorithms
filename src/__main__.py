import methods.utils
import methods.fitness_score
import methods.selection_methods
import methods.reproduction_tools
import os

from src.methods.reproduction_tools import single_crossover, parent_pairing

LARGE_SCALE_PATH_PREFIX = "../dane AG 2/large_scale/"
LOW_DIMENSIONAL_PATH_PREFIX = "../dane AG 2/low-dimensional/"
MAX_WEIGHT = 1000

ITEMS_VALUE_WEIGHT = methods.utils.load_data(
    LOW_DIMENSIONAL_PATH_PREFIX + "f3_l-d_kp_4_20"
)

methods.utils.create_population_file(int(4), len(ITEMS_VALUE_WEIGHT), 100, 2137)
fitness = methods.fitness_score.calc_fitness_score(ITEMS_VALUE_WEIGHT, MAX_WEIGHT)
parent_pool = methods.selection_methods.fitness_proportionate_selection(
    fitness, parent_group_size=4, seed=2137
)
parent_pairs = parent_pairing(parent_pool)
single_crossover(parent_pairs, seed=2137, cross_propab=1)


files = ["population.dat", "population.json"]
for file in files:
    if os.path.exists(file):
        os.remove(file)
        print(f"removed {file}")
