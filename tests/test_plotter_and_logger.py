"""Smoke tests for Plotter and logger initialization."""

import logging

import matplotlib
import src.methods.logging_library as logging_library
from src.classes.Plotter import Plotter

matplotlib.use("Agg")


def test_plotter_creates_plot_file(
    experiment_config_factory, test_only_pathresolver
) -> None:
    """Ensure the Plotter saves a PNG when data and optimum are present."""
    test_only_pathresolver.initialize("f_plotter_case")
    config = experiment_config_factory(
        population_size=4,
        generations=2,
        max_weight=10,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.5,
        mutation_probability=0.1,
        penalty_multiplier=1.0,
    )

    optimum_dir = test_only_pathresolver.get_optimum_path()
    optimum_dir.mkdir(parents=True, exist_ok=True)
    (optimum_dir / config.data_filename).write_text("25\n")

    output_dir = test_only_pathresolver.get_output_path()
    csv_path = output_dir / f"{test_only_pathresolver.filename_constant}.csv"
    csv_path.write_text(
        "iteration,worst_fitness,best_fitness,avg_fitness\n" "0,1,5,3\n" "1,2,6,4\n"
    )

    plotter = Plotter(test_only_pathresolver, config)
    plotter.performance_and_correctness()

    plot_path = test_only_pathresolver.get_plot_path() / "best_fitness_v_optimal.png"
    assert plot_path.exists()


def test_logger_initialization_creates_log_file(
    experiment_config_factory, test_only_pathresolver
) -> None:
    """Initialize logger and confirm it writes the expected log file."""
    test_only_pathresolver.initialize("f_logger_case")
    config = experiment_config_factory(
        population_size=4,
        generations=2,
        max_weight=10,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.5,
        mutation_probability=0.1,
        penalty_multiplier=1.0,
        exp_identifier=7,
        log_level="INFO",
    )

    main_logger = logging.getLogger("GA experiment run")
    for handler in list(main_logger.handlers):
        main_logger.removeHandler(handler)
        handler.close()

    logger = logging_library.initialize(config, test_only_pathresolver)
    logger.info("hello logger")
    for handler in logger.logger.handlers:
        handler.flush()

    log_file = (
        test_only_pathresolver.get_logging_path()
        / f"runtime_experiment_{config.experiment_identifier}.log"
    )
    assert log_file.exists()
    assert "hello logger" in log_file.read_text()


def test_logger_initialize_sets_handlers_and_level(
    experiment_config_factory, test_only_pathresolver
) -> None:
    """Initialization attaches handlers once and applies the configured level."""
    test_only_pathresolver.initialize("f_logger_init_case")
    config = experiment_config_factory(
        population_size=4,
        generations=2,
        max_weight=10,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.5,
        mutation_probability=0.1,
        penalty_multiplier=1.0,
        exp_identifier=9,
        log_level="DEBUG",
    )

    main_logger = logging.getLogger("GA experiment run")
    for handler in list(main_logger.handlers):
        main_logger.removeHandler(handler)
        handler.close()

    adapter = logging_library.initialize(config, test_only_pathresolver)

    assert adapter.extra["exp_id"] == config.experiment_identifier
    assert adapter.logger.getEffectiveLevel() == logging.DEBUG
    assert len(adapter.logger.handlers) == 2


def test_logger_initialize_falls_back_on_level_error(
    experiment_config_factory, test_only_pathresolver, monkeypatch
) -> None:
    """If level resolution fails, initialization should fall back to INFO."""
    test_only_pathresolver.initialize("f_logger_level_case")
    config = experiment_config_factory(
        population_size=4,
        generations=2,
        max_weight=10,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.5,
        mutation_probability=0.1,
        penalty_multiplier=1.0,
        exp_identifier=12,
        log_level="INVALID",
    )

    main_logger = logging.getLogger("GA experiment run")
    for handler in list(main_logger.handlers):
        main_logger.removeHandler(handler)
        handler.close()

    adapter = logging_library.initialize(config, test_only_pathresolver)
    assert adapter.logger.getEffectiveLevel() == logging.INFO


def test_log_generation_writes_expected_summary(
    experiment_config_factory, test_only_pathresolver
) -> None:
    """log_generation should persist a readable summary into the log file."""
    test_only_pathresolver.initialize("f_logger_generation_case")
    config = experiment_config_factory(
        population_size=4,
        generations=2,
        max_weight=10,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.5,
        mutation_probability=0.1,
        penalty_multiplier=1.0,
        exp_identifier=11,
        log_level="INFO",
    )

    main_logger = logging.getLogger("GA experiment run")
    for handler in list(main_logger.handlers):
        main_logger.removeHandler(handler)
        handler.close()

    logger = logging_library.initialize(config, test_only_pathresolver)
    logging_library.log_generation(
        logger,
        best_idx=2,
        best_score=15,
        weight=7,
        iteration=3,
        repetitions=1,
    )
    for handler in logger.logger.handlers:
        handler.flush()

    log_file = (
        test_only_pathresolver.get_logging_path()
        / f"runtime_experiment_{config.experiment_identifier}.log"
    )
    content = log_file.read_text()
    assert "Generation 3" in content
    assert "Fitness of best individual: 15" in content
    assert "There were 1 individuals with same scores" in content


def test_log_generation_skips_repetition_message_when_zero(
    experiment_config_factory, test_only_pathresolver
) -> None:
    """When repetitions are zero the repetition line should not be emitted."""
    test_only_pathresolver.initialize("f_logger_generation_zero_rep")
    config = experiment_config_factory(
        population_size=4,
        generations=2,
        max_weight=10,
        selection_type="roulette",
        crossover_type="one",
        crossover_probability=0.5,
        mutation_probability=0.1,
        penalty_multiplier=1.0,
        exp_identifier=13,
        log_level="INFO",
    )

    main_logger = logging.getLogger("GA experiment run")
    for handler in list(main_logger.handlers):
        main_logger.removeHandler(handler)
        handler.close()

    logger = logging_library.initialize(config, test_only_pathresolver)
    logging_library.log_generation(
        logger,
        best_idx=0,
        best_score=5,
        weight=3,
        iteration=1,
        repetitions=0,
    )
    for handler in logger.logger.handlers:
        handler.flush()

    log_file = (
        test_only_pathresolver.get_logging_path()
        / f"runtime_experiment_{config.experiment_identifier}.log"
    )
    content = log_file.read_text()
    assert "Generation 1" in content
    assert "There were" not in content
