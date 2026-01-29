"""Tests for CSV writer utilities."""

from pathlib import Path

import pytest
from src.ga_core.output.csv_writer import (
    CsvGenericOutput,
    CsvHandler,
    ExperimentCsvOutput,
)


def test_csv_handler_opens_and_writes(tmp_path: Path) -> None:
    handler = CsvHandler(output_path=tmp_path, exp_filename="exp1")

    handler.open()
    handler.writer.writerow(["col1", "col2"])
    handler.writer.writerow([1, 2])
    handler.close()

    csv_path = tmp_path / "exp1.csv"
    assert csv_path.exists()
    content = csv_path.read_text().strip().splitlines()
    assert content == ["col1,col2", "1,2"]


def test_generic_output_writes_headers_and_rows(tmp_path: Path) -> None:
    handler = CsvHandler(output_path=tmp_path, exp_filename="exp2")
    writer = CsvGenericOutput(handler)
    with writer:
        writer.init_csv(headers=["a", "b"])
        writer.write_csv_row(1, 2)
        writer(3, 4)

    csv_path = tmp_path / "exp2.csv"
    lines = csv_path.read_text().strip().splitlines()
    assert lines == ["a,b", "1,2", "3,4"]


def test_generic_output_raises_when_row_length_mismatch(tmp_path: Path) -> None:
    handler = CsvHandler(output_path=tmp_path, exp_filename="exp3")
    writer = CsvGenericOutput(handler)
    with writer:
        writer.init_csv(headers=["a", "b"])
        with pytest.raises(ValueError):
            writer.write_csv_row(1)


def test_experiment_csv_output_initializes_metadata(tmp_path: Path) -> None:
    writer = ExperimentCsvOutput.from_input(exp_filename="exp4", output_path=tmp_path)
    with writer:
        writer.init_experiment_csv(
            job_id="job1",
            data_filename="data.csv",
            population_size=10,
            generations=5,
            max_weight=100,
            seed=1,
            selection_type="tournament",
            crossover_type="one",
            crossover_probability=0.6,
            mutation_probability=0.1,
            penalty=2.0,
            experiment_identifier="exp",
            log_level="INFO",
            tournament_size=3,
        )
        writer.write_iteration(
            iteration=1,
            best_fitness=10,
            best_weight=5,
            avg_fitness=7.5,
            worst_fitness=2,
            worst_weight=3,
            identical_best_count=0,
            genome="1010",
        )

    csv_path = tmp_path / "exp4.csv"
    lines = [line for line in csv_path.read_text().splitlines() if line]
    assert lines[0].startswith("#job_id,job1")
    assert "iteration,best_fitness" in lines[-2]
    assert lines[-1].startswith("1,10,5")


def test_experiment_csv_output_requires_selection_specific_fields(
    tmp_path: Path,
) -> None:
    writer = ExperimentCsvOutput.from_input(exp_filename="exp5", output_path=tmp_path)
    with writer:
        with pytest.raises(ValueError):
            writer.init_experiment_csv(
                job_id="job1",
                data_filename="data.csv",
                population_size=10,
                generations=5,
                max_weight=100,
                seed=1,
                selection_type="tournament",
                crossover_type="one",
                crossover_probability=0.6,
                mutation_probability=0.1,
                penalty=2.0,
            )
