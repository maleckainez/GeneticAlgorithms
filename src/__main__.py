"""Entry point for genetic algorithm knapsack solver."""

import csv
from pathlib import Path

from src.ga_core.config.experiment_config import ExperimentConfig
from src.ga_core.config.input_config_scheme import EncodingType
from src.ga_core.engine.evolution_engine import EvolutionEngine, GenerationStats
from src.ga_core.engine.population.manager import PopulationManager
from src.ga_core.engine.strategies.executor import StrategyExecutor
from src.ga_core.io.loader import load_experiment_data, load_yaml_config
from src.ga_core.storage.directory_utils import ensure_layout_paths
from src.ga_core.storage.experiment_storage import ExperimentStorage


class SimpleStorageLayout:
    """Simple storage layout implementation."""

    def __init__(self, root: Path):
        """Initialize layout."""
        self._root = root
        self._temp = root / "temp"
        self._output = root / "output"
        self._logs = root / "logs"
        self._plots = root / "output" / "plots"

    @property
    def temp(self) -> Path:
        """Temp directory."""
        return self._temp

    @property
    def output(self) -> Path:
        """Output directory."""
        return self._output

    @property
    def logs(self) -> Path:
        """Logs directory."""
        return self._logs

    @property
    def plots(self) -> Path:
        """Plots directory."""
        return self._plots


def main():
    """Run genetic algorithm from config.yaml."""
    # Load config
    config_path = Path("config.yaml")
    input_config = load_yaml_config(config_path)

    # Load data
    data_path = Path("dane AG 2") / "low-dimensional" / input_config.data.data_filename
    items_data = load_experiment_data(data_path)

    # Create runtime config
    config = ExperimentConfig(
        input=input_config,
        job_id=input_config.experiment.identifier,
        root_path=Path.cwd(),
    )

    # Setup storage
    layout = SimpleStorageLayout(root=config.root_path)
    ensure_layout_paths(layout)
    storage = ExperimentStorage(
        layout=layout,
        job_id=config.job_id,
        data_file_name=input_config.data.data_filename,
    )

    # Create RNG
    rng = config.create_rng()

    # Create fitness function wrapper with items data
    from src.ga_core.strategies.binary.fitness_score import fitness_calculation

    def fitness_fn_wrapper(population):
        return fitness_calculation(
            max_weight=input_config.data.max_weight,
            penalty_factor=config.penalty_multiplier,
            population=population,
            batch_size=config.stream_batch_size,
            value_arr=items_data[:, 0],
            weight_arr=items_data[:, 1],
        )

    # Create strategy executor manually with wrapped fitness
    from src.ga_core.strategies.factory import (
        create_pairing_fn,
        create_reproduction_executor,
        create_selection_fn,
    )

    selection_fn = create_selection_fn(
        selection_type=config.selection_type,
        population_size=config.population_size,
        tournament_size=config.tournament_size or 3,
        selection_pressure=config.selection_pressure or 1.5,
    )

    pairing_fn = create_pairing_fn("random")

    reproduction_executor = create_reproduction_executor(
        encoding=EncodingType.BINARY,
        crossover_type=config.crossover_type,
        crossover_probability=config.crossover_probability,
        mutation_probability=config.mutation_probability,
    )

    strategy_executor = StrategyExecutor(
        selection_fn=selection_fn,
        pairing_fn=pairing_fn,
        reproduction_executor=reproduction_executor,
        fitness_fn=fitness_fn_wrapper,
        population_size=config.population_size,
    )

    # Create population manager
    population_manager = PopulationManager(
        population_size=config.population_size,
        genome_length=items_data.shape[0],
        stream_batch_size=config.stream_batch_size,
        storage=storage,
        rng=rng,
        overweight_probability=config.estimate_overweight_probability(
            items_data[:, 1].sum()
        ),
        commit_mode=config.commit_mode,
    )

    # Initialize population
    population_manager.initialize_population()
    population_manager.initialize_children()

    # Create evolution engine
    engine = EvolutionEngine(
        strategy_executor=strategy_executor,
        config=config,
        population_manager=population_manager,
        rng=rng,
    )

    # Setup CSV logging
    csv_path = layout.output / f"{config.job_id}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["iteration", "best_fitness", "best_weight", "avg_fitness"])

    def log_generation(stats: GenerationStats):
        """Log generation stats to CSV."""
        csv_writer.writerow(
            [
                stats.iteration,
                stats.best_fitness,
                stats.best_weight,
                stats.avg_fitness,
            ]
        )
        csv_file.flush()

    # Run evolution
    print(f"Starting: {config.generations} gens, " f"pop {config.population_size}")

    engine.run_evolution(
        population=population_manager.population,
        children=population_manager.children,
        generations=config.generations,
        on_generation=log_generation,
    )

    # Finalize
    engine.finalize()
    csv_file.close()

    print(f"Evolution complete. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
