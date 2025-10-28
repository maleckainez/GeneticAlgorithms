import methods.utils
import methods.fitness_score
import methods.selection_methods


LARGE_SCALE_PATH_PREFIX = "../dane AG 2/large_scale/"
LOW_DIMENSIONAL_PATH_PREFIX = "../dane AG 2/low-dimensional/"
MAX_WEIGHT = 5000

ITEMS_VALUE_WEIGHT = methods.utils.load_data(
    LARGE_SCALE_PATH_PREFIX + "knapPI_1_10000_1000_1"
)

methods.utils.create_population_file(10000, len(ITEMS_VALUE_WEIGHT), 500, 2137)
fitness = methods.fitness_score.calc_fitness_score(ITEMS_VALUE_WEIGHT, MAX_WEIGHT)
methods.selection_methods.fitness_proportionate_selection(fitness, parent_group_size=5000, seed=2137)