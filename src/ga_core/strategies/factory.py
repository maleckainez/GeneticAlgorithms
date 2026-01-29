"""Strategy factory for selecting genetic algorithm components.

Provides encoding-agnostic facade for creating selection functions,
fitness evaluators, and reproduction executors based on configuration.
"""

from src.ga_core.config.input_config_scheme import (
    CrossoverType,
    EncodingType,
    SelectionType,
)
from src.ga_core.strategies.elitism import select_elites
from src.ga_core.strategies.parent_pairing import (
    random_pairing,
    sequential_pairing,
)
from src.ga_core.strategies.protocols import (
    ElitismFn,
    FitnessFn,
    PairingFn,
    ReproductionExecutor,
    SelectionFn,
)
from src.ga_core.strategies.selection_methods import (
    linear_rank_selection,
    roulette_selection,
    tournament_selection,
)


def create_selection_fn(
    selection_type: SelectionType,
    population_size: int,
    tournament_size: int = 3,
    selection_pressure: float = 1.5,
) -> SelectionFn:
    """Create selection function based on configuration.

    Args:
        selection_type: Type of selection method.
        population_size: Size of population.
        tournament_size: Tournament size (for tournament selection).
        selection_pressure: Selection pressure (for rank selection).

    Returns:
        Selection function matching SelectionFn protocol.
    """
    if selection_type == SelectionType.ROULETTE:
        return lambda fitness_arr, pop_size, rng, *args, **kwargs: roulette_selection(
            fitness_arr, pop_size, rng
        )
    elif selection_type == SelectionType.TOURNAMENT:
        return lambda fitness_arr, pop_size, rng, *args, **kwargs: tournament_selection(
            fitness_arr, pop_size, rng, tournament_size
        )
    elif selection_type == SelectionType.LINEAR_RANK:
        return (
            lambda fitness_arr, pop_size, rng, *args, **kwargs: linear_rank_selection(
                fitness_arr, pop_size, rng, selection_pressure
            )
        )
    else:
        raise ValueError(f"Unknown selection type: {selection_type}")


def create_pairing_fn(pairing_strategy: str = "random") -> PairingFn:
    """Create parent pairing function.

    Args:
        pairing_strategy: Strategy for pairing parents ("random" or "sequential").

    Returns:
        Pairing function matching PairingFn protocol.
    """
    if pairing_strategy == "random":
        return random_pairing
    elif pairing_strategy == "sequential":
        return sequential_pairing
    else:
        raise ValueError(f"Unknown pairing strategy: {pairing_strategy}")


def create_elitism_fn() -> ElitismFn:
    """Create elitism function.

    Returns:
        Elitism function matching ElitismFn protocol.
    """
    return select_elites


def create_reproduction_executor(
    encoding: EncodingType,
    crossover_type: CrossoverType,
    crossover_probability: float,
    mutation_probability: float,
) -> ReproductionExecutor:
    """Create reproduction executor based on encoding and crossover type.

    Args:
        encoding: Encoding type (currently only BINARY supported).
        crossover_type: Type of crossover operation.
        crossover_probability: Probability of crossover per pair.
        mutation_probability: Probability of mutation per gene.

    Returns:
        Reproduction executor matching ReproductionExecutor protocol.

    Raises:
        ValueError: If encoding type is not supported.
    """
    if encoding == EncodingType.BINARY:
        from src.ga_core.strategies.binary.reproduction_builder import (
            create_binary_reproduction_executor,
        )

        return create_binary_reproduction_executor(
            crossover_type=crossover_type,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
        )
    else:
        raise ValueError(f"Unsupported encoding type: {encoding}")


def create_fitness_fn(encoding: EncodingType) -> FitnessFn:
    """Create fitness evaluation function based on encoding type.

    Args:
        encoding: Encoding type (currently only BINARY supported).

    Returns:
        Fitness function matching FitnessFn protocol.

    Raises:
        ValueError: If encoding type is not supported.
    """
    if encoding == EncodingType.BINARY:
        from src.ga_core.strategies.binary.fitness_score import fitness_calculation

        return fitness_calculation
    else:
        raise ValueError(f"Unsupported encoding type: {encoding}")
