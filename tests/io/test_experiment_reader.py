"""Tests for ExperimentReader."""

from pathlib import Path

import pandas as pd
from src.ga_core.io.experiment_reader import ExperimentReader


def test_experiment_reader_loads_results_and_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "#job_id,exp-1\n"
        "#seed,123\n"
        "\n"
        "iteration,best_fitness,avg_fitness,worst_fitness\n"
        "0,1,0.5,0\n"
        "1,2,1.5,1\n"
    )

    reader = ExperimentReader(csv_path)

    results = reader.load_results()
    metadata = reader.load_metadata()

    pd.testing.assert_frame_equal(
        results,
        pd.DataFrame(
            {
                "iteration": [0, 1],
                "best_fitness": [1, 2],
                "avg_fitness": [0.5, 1.5],
                "worst_fitness": [0, 1],
            }
        ),
    )
    assert metadata == {"job_id": "exp-1", "seed": "123"}
