"""Tests for in-memory population memmap creation and cleanup class."""

from collections.abc import Callable

from numpy.testing import assert_array_equal
from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver
from src.classes.PopulationHandler import PopulationHandler


def test_correct_population_creation(
    test_only_pathresolver: PathResolver,
    experiment_config_factory: Callable[..., ExperimentConfig],
):
    config = experiment_config_factory(
        population_size=6,
        generations=2,
        max_weight=10,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.5,
        mutation_probability=0.0,
        penalty_multiplier=1.0,
    )
    genome_length = 10
    weight_sum = 100

    population_manager = PopulationHandler(
        config=config,
        paths=test_only_pathresolver,
        genome_length=genome_length,
        filename_constant=test_only_pathresolver.filename_constant,
        weight_sum=weight_sum,
    )

    pop_handle = population_manager.get_pop_handle()
    pop_config = population_manager.get_pop_config()
    assert pop_handle is not None
    assert_array_equal(pop_handle, population_manager.pop_handle)
    assert pop_config is not None
    assert pop_config == population_manager.pop_config
    population_manager.close()
    population_manager.close()
    assert population_manager.get_pop_handle() is None
    population_manager.open_pop()
    population_manager.open_pop()
    assert population_manager.get_pop_handle() is not None
