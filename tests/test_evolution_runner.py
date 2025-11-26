"""Top-level smoke tests for the EvolutionRunner entry point."""

import csv
import logging

import pytest
from src.classes.EvolutionRunner import EvolutionRunner
from src.classes.PathResolver import PathResolver


def test_evolution_runner_rejects_zero_generations() -> None:
    """Constructing runner with generations=0 should raise configuration error."""
    with pytest.raises(ValueError, match="Generations must be greater than 1"):
        EvolutionRunner(
            {
                "data_filename": "f_dummy.txt",
                "population_size": 2,
                "generations": 0,
                "max_weight": 10,
                "seed": 123,
                "selection_type": "roulette",
                "crossover_type": "one",
                "crossover_probability": 0.5,
                "mutation_probability": 0.1,
                "penalty": 1.0,
                "experiment_identifier": 1,
                "log_level": "INFO",
            }
        )


def test_evolution_runner_invalid_selection_type(tmp_path, monkeypatch) -> None:
    """Runner should validate selection strategy before entering main loop."""
    monkeypatch.setattr(PathResolver, "PROJECT_ROOT", tmp_path)
    data_dir = tmp_path / "dane AG 2" / "low-dimensional"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "f_dummy.txt").write_text("10 5\n")

    main_logger = logging.getLogger("GA experiment run")
    for handler in list(main_logger.handlers):
        main_logger.removeHandler(handler)
        handler.close()

    with pytest.raises(ValueError, match="Invalid selection method"):
        EvolutionRunner(
            {
                "data_filename": "f_dummy.txt",
                "population_size": 2,
                "generations": 1,
                "max_weight": 10,
                "seed": 123,
                "selection_type": "invalid",
                "crossover_type": "one",
                "crossover_probability": 0.5,
                "mutation_probability": 0.1,
                "penalty": 1.0,
                "experiment_identifier": 2,
                "log_level": "INFO",
            }
        )


def test_evolution_runner_evolve_writes_iterations(tmp_path, monkeypatch) -> None:
    """Full evolve run (1 generation) writes iteration rows to CSV output."""
    monkeypatch.setattr(PathResolver, "PROJECT_ROOT", tmp_path)
    data_dir = tmp_path / "dane AG 2" / "low-dimensional"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "f_dummy.txt").write_text("10 5\n8 4\n")
    monkeypatch.setattr(
        "src.classes.Plotter.Plotter.performance_and_correctness", lambda self: None
    )
    monkeypatch.setattr("src.methods.utils.final_screen", lambda: None)

    main_logger = logging.getLogger("GA experiment run")
    for handler in list(main_logger.handlers):
        main_logger.removeHandler(handler)
        handler.close()

    runner = EvolutionRunner(
        {
            "data_filename": "f_dummy.txt",
            "population_size": 2,
            "generations": 1,
            "max_weight": 10,
            "seed": 123,
            "selection_type": "roulette",
            "crossover_type": "one",
            "crossover_probability": 0.5,
            "mutation_probability": 0.1,
            "penalty": 1.0,
            "experiment_identifier": 3,
            "log_level": "INFO",
        }
    )

    runner.evolve()

    csv_path = runner.paths.get_output_path() / f"{runner.paths.filename_constant}.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    iteration_rows = [row for row in rows if row and row[0].isdigit()]

    assert {row[0] for row in iteration_rows} == {"0", "1"}
