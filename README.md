![GitHub Tag](https://img.shields.io/github/v/tag/maleckainez/GeneticAlgorithms?include_prereleases)

[![Build Status](https://github.com/maleckainez/GeneticAlgorithms/actions/workflows/ci.yml/badge.svg)](https://github.com/maleckainez/GeneticAlgorithms/actions/workflows/ci.yml)
![Endpoint Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmaleckainez%2FGeneticAlgorithms%2Frefs%2Fheads%2Fcoverage_score%2F.github%2Fcoverage.json)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# Genetic Algorithms – Knapsack Problem

Implementation of genetic algorithms for classic knapsack benchmark instances (from the `dane AG 2` dataset).
The project allows running experiments with different configurations, logging results, and comparing them against known optimal solutions.

## Requirements

- **Python ≥ 3.10**
- OS: Windows / Linux / macOS
- Installed automatically via `pip install .`:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `PyYAML`
- Development (optional):
  - `pytest`
  - `black`
  - `pre-commit`
  - `mypy`
  - `ruff`
  - `coverage`


## Installation

These instructions assume that you have Python ≥ 3.10 and `git` installed.

1. Clone the repository:

   ```bash
   git clone https://github.com/maleckainez/GeneticAlgorithms.git
   cd GeneticAlgorithms
   ```

3. Create and activate a virtual environment:

    ```bash
    python -m venv .venv
    ```
    ```bash
    # Windows:
    .venv\Scripts\activate
    ```
    ```bash
    # Linux/macOS:
    . .venv/bin/activate
    ```
4. Install the project with its dependencies:

   ```bash
   pip install .
   ```


## Development

This section describes how to set up a full development environment, run tests, and execute linters and type checkers.

### 1. Setup

Install the project in editable mode together with all development tools:

```
pip install -e '.[dev]'
```

Then install pre-commit hooks:

```
pre-commit install
```

### 2. Running tests

From the project root:

```
pytest
```

### 3. Running linters and type checks

Code style and static analysis:

```
ruff check src tests
black --check src tests
mypy src
```

### 4. Test coverage

To measure code coverage:

```
coverage run -m pytest
coverage report
```


## Dataset: `dane AG 2`

The project uses knapsack benchmark instances located in:

```
dane AG 2/
├── large_scale/               # large knapsack instances (knapPI_...)
├── large_scale-optimum/       # optimum values for large_scale
├── low-dimensional/           # smaller instances (f...)
└── low-dimensional-optimum/   # optimum values for low-dimensional
```

The `PathResolver` module (`src/classes/PathResolver.py`) assumes:

- files starting with **`knap`** are stored in `dane AG 2/large_scale`
- files starting with **`f`** are stored in `dane AG 2/low-dimensional`

If you use your own dataset, it must:

- contain two integers per line (`value weight`)
- follow the same folder structure; or
- require modification of `PathResolver`.

## Configuration (`config.yaml`)

All experiment parameters are defined in `config.yaml` in the repository root.

Example:

```yaml
data:
  filename: "knapPI_1_10000_1000_1"
  max_weight: 1000

population:
  size: 10000
  generations: 4000
  stream_batch_size: 500

selection:
  type: "roulette"       # [roulette, tournament, rank]
  selection_pressure: 2  # only for rank (1–2)

genetic_operators:
  crossover_type: "two"         # [one, two]
  crossover_probability: 0.75   # 0–1
  mutation_probability: 0.0002  # 0–1
  penalty_multiplier: 1         #penalty for exceeding max weight; 0 is a special value that sets fitness to 0 for overweight individuals, any other value acts as a penalty multiplier

experiment:
  seed: 2137                    # RNG seed (for reproducibility)
  identifier: 1                 # experiment ID (used in file names)
  log_level: INFO               # [DEBUG, INFO, WARNING, ERROR, CRITICAL]
```

### Key fields for reproducibility

- **`experiment.seed`** – controls randomness
  Same seed ⇒ same random sequence (assuming identical code and library versions).
- **`data.filename`** – selects which knapsack instance is used.
- **`population.*`, `selection.*`, `genetic_operators.*`** – must match exactly to reproduce a specific experiment.

## Running Experiments

From the repository root (where `pyproject.toml` and `config.yaml` are located):

```bash
python -m src
```

The script:

- loads configuration from `config.yaml`
- creates:
  - `ExperimentConfig`
  - `PathResolver`
  - `EvolutionRunner`
- runs the evolutionary process
- saves output to:

```
run_output/<experiment-name>/
├── logs/
├── output/
└── plots/
```

The plot `best_fitness_v_optimal.png` compares the best fitness found by the GA with the known optimal value.

Experiment naming and output directory structure are defined in:

- `src/methods/experiment_defining_tools.py`
- `src/classes/PathResolver.py`


## License

> This project is available under the MIT License.

