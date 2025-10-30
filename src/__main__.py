import methods.utils
import methods.fitness_score
import methods.selection_methods
import os

LARGE_SCALE_PATH_PREFIX = "../dane AG 2/large_scale/"
LOW_DIMENSIONAL_PATH_PREFIX = "../dane AG 2/low-dimensional/"
MAX_WEIGHT = 10

ITEMS_VALUE_WEIGHT = methods.utils.load_data(
    LARGE_SCALE_PATH_PREFIX + "knapPI_2_200_1000_1"
)

methods.utils.create_population_file(int(10), len(ITEMS_VALUE_WEIGHT), 100, 2137)
fitness = methods.fitness_score.calc_fitness_score(ITEMS_VALUE_WEIGHT, MAX_WEIGHT)
print(f"fitenss: {fitness}")
methods.selection_methods.fitness_proportionate_selection(fitness, parent_group_size=10, seed=2137)
files = ["population.dat", "population.json"]
for file in files:
    if os.path.exists(file):
        os.remove(file)
        print(f"removed {file}")