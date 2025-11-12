import numpy as np

from src.classes.ExperimentConfig import ExperimentConfig


def roulette_selection(fitness_arr: np.ndarray, config: ExperimentConfig) -> list[int]:
    fitness_sum = fitness_arr[:, 0].sum()
    if fitness_sum == 0:
        fitness_arr[:, 0] = 1
        fitness_sum = fitness_arr.shape[0]
    fitness_proportionate = fitness_arr[:, 0] / fitness_sum
    proportionate_cfd = np.cumsum(fitness_proportionate.flatten())
    proportionate_cfd[-1] = 1
    r = config.rng.random(config.population_size)
    return np.searchsorted(proportionate_cfd, r).tolist()


def tournament_selection(
    fitness_arr: np.ndarray,
    config: ExperimentConfig,
) -> list[int]:
    tournament_size = 5
    rng = config.rng
    fitness_score_arr = fitness_arr[:, 0]
    selected_parents = []
    # takes up chunk of population, checks best idx of fitness
    for i in range(config.population_size):
        gladiators = rng.choice(
            config.population_size,
            size=tournament_size,
            replace=False,
        )
        winner = int(np.argmax(fitness_score_arr[gladiators]))
        selected_parents.append(int(gladiators[winner]))
    return selected_parents

def _tournament_tie_breaker(gladiators: list[int],
                            fitness_arr: np.ndarray):
    raise NotImplementedError()

def linear_rank_selection(
        fitness_arr: np.ndarray,
        config: ExperimentConfig
):  
    selective_pressure = 2
    SP = selective_pressure
    #Consider Nind the number of individuals in the population, Pos the position of an individual in this population 
    # (least fit individual has Pos=1, the fittest individual Pos=Nind) and SP the selective pressure. 
    # The fitness value for an individual is calculated as:
    # Fitess(pos) = 2 - SP + 2*(SP-1)*((pos-1)/Nind-1)
    print(_calc_pressured_fitness(SP, fitness_arr))

def _calc_pressured_fitness(SP:float,fitness_arr:np.ndarray):
    ascending_sorted_idxes = np.lexsort((-fitness_arr[:,1], fitness_arr[:,0]))
    length = len(ascending_sorted_idxes)
    positions = np.arange(1, length+1)
    pressured_fitness = 2-SP + 2*(SP-1) *((positions-1)/(length-1))
    return pressured_fitness