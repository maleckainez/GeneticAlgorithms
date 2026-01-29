"""Factory for creating strategy executors from configuration."""

from src.ga_core.config.experiment_config import ExperimentConfig
from src.ga_core.config.input_config_scheme import EncodingType
from src.ga_core.engine.strategies.executor import StrategyExecutor
from src.ga_core.strategies.factory import (
    create_fitness_fn,
    create_pairing_fn,
    create_reproduction_executor,
    create_selection_fn,
)


def create_strategy_executor(
    config: ExperimentConfig,
    encoding: EncodingType = EncodingType.BINARY,
    pairing_strategy: str = "random",
) -> StrategyExecutor:
    """Create strategy executor from configuration.

    Factory function that assembles all strategies based on configuration
    parameters and returns a ready-to-use executor.

    Args:
        config: Experiment configuration with strategy parameters.
        encoding: Genome encoding type (default: binary).
        pairing_strategy: Parent pairing strategy (default: random).

    Returns:
        Configured StrategyExecutor instance.
    """
    # Create selection function
    selection_fn = create_selection_fn(
        selection_type=config.selection_type,
        population_size=config.population_size,
        tournament_size=config.tournament_size or 3,
        selection_pressure=config.selection_pressure or 1.5,
    )

    # Create pairing function
    pairing_fn = create_pairing_fn(pairing_strategy)

    # Create reproduction executor
    reproduction_executor = create_reproduction_executor(
        encoding=encoding,
        crossover_type=config.crossover_type,
        crossover_probability=config.crossover_probability,
        mutation_probability=config.mutation_probability,
    )

    # Create fitness function
    fitness_fn = create_fitness_fn(encoding)

    # Assemble executor
    return StrategyExecutor(
        selection_fn=selection_fn,
        pairing_fn=pairing_fn,
        reproduction_executor=reproduction_executor,
        fitness_fn=fitness_fn,
        population_size=config.population_size,
    )
