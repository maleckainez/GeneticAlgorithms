import methods.utils
import methods.fitness_score


LARGE_SCALE_PATH_PREFIX = "../dane AG 2/large_scale/"
LOW_DIMENSIONAL_PATH_PREFIX = "../dane AG 2/low-dimensional/"
MAX_WEIGHT = 1000

ITEMS_VALUE_WEIGHT = methods.utils.load_data(
    LOW_DIMENSIONAL_PATH_PREFIX + "f1_l-d_kp_10_269"
)
methods.utils.create_population_file(1000, len(ITEMS_VALUE_WEIGHT), 100, 2137)
print(methods.fitness_score.calc_fitness_score(ITEMS_VALUE_WEIGHT, MAX_WEIGHT))
