![GitHub Tag](https://img.shields.io/github/v/tag/maleckainez/GeneticAlgorithms?include_prereleases)

[![Build Status](https://github.com/maleckainez/GeneticAlgorithms/actions/workflows/ci.yml/badge.svg)](https://github.com/maleckainez/GeneticAlgorithms/actions/workflows/ci.yml)
![Endpoint Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmaleckainez%2FGeneticAlgorithms%2Frefs%2Fheads%2Fcoverage_score%2F.github%2Fcoverage.json&cacheSeconds=300)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python library and web API for genetic algorithm experiments on optimization problems, primarily focused on knapsack problem benchmarks.

## Project Structure

This repository contains:

- **`src/ga_core/`** – Core genetic algorithm library (publishable to PyPI)
- **`src/api/`** – FastAPI web service for running GA experiments
- **`dane AG 2/`** – Benchmark knapsack problem instances
- **`tests/`** – Test suite

---

## Installation

### Library Only (for use in your code)

```bash
pip install genetic-algorithms
```

### Full Development Setup

```bash
git clone https://github.com/maleckainez/GeneticAlgorithms.git
cd GeneticAlgorithms
python -m venv .venv
# Linux:
source .venv/bin/activate  
# Windows: 
.venv\Scripts\activate
# MacOS:
. .venv/bin/activate
pip install -e ".[dev,api]"
```

**Install options:**
- Base: `pip install .` – Core library only
- `.[api]` – Include FastAPI dependencies
- `.[dev]` – Include testing & linting tools
- `.[all]` - Install all dependencies and tools
---

## Library Usage (`ga_core`)

### Quick Start

```python
from pathlib import Path
from src.ga_core import ExperimentConfig, InputConfig, ExperimentStorage
from src.ga_core.engine import EvolutionEngine
from src.ga_core.io import load_yaml_config, load_experiment_data

# Load configuration
config_path = Path("config.yaml")
input_config = load_yaml_config(config_path)

# Load knapsack data
data_path = Path("dane AG 2/low-dimensional/f1_l-d_kp_10_269")
items_data = load_experiment_data(data_path)

# Create experiment config
config = ExperimentConfig(
    input=input_config,
    job_id="experiment-001",
    root_path=Path.cwd()
)

# Setup storage
from src.ga_core.storage import SimpleStorageLayout
layout = SimpleStorageLayout(root=config.root_path)
storage = ExperimentStorage(layout=layout, job_id=config.job_id)

# Run evolution
engine = EvolutionEngine(config=config, storage=storage, items_data=items_data)
engine.run()
```

### Configuration (`config.yaml`)

```yaml
data:
  data_filename: "f1_l-d_kp_10_269"
  max_weight: 269

population:
  size: 100
  generations: 500
  stream_batch_size: 50
  commit_mode: "swap"  # or "copy"

selection:
  type: "roulette"  # roulette, tournament, linear_rank
  selection_pressure: null  # for rank selection: 1.0-2.0
  tournament_size: null     # for tournament: int

genetic_operators:
  crossover_type: "two_point"  # one_point, two_point
  crossover_probability: 0.75
  mutation_probability: 0.01
  penalty_multiplier: 1.0
  strict_weight_constraints: false

experiment:
  seed: 42
  identifier: "exp-001"
  log_level: "INFO"
```

---

## Web API

### Running the API

```bash
# Development mode
fastapi dev src/api/main.py --reload --reload-dir src/

# Production mode
fastapi run src/api/main.py
```

API will be available at:
-  http://localhost:8000

API Docs at:
-  http://localhost:8000/docs

### API Endpoints

**Health Check**
```bash
GET /health
```

**Submit GA Job**
```bash
POST /backend/run/ga
Content-Type: application/json

{
  "data": {"data_filename": "f1_l-d_kp_10_269", "max_weight": 269},
  "population": {"size": 100, "generations": 500, ...},
  ...
}

Response: {"job_id": "job-uuid", "status": "pending"}
```

**Check Job Status**
```bash
GET /backend/status/{job_id}

Response: {"job_id": "uuid", "status": "running"}
```

**List Jobs by Status**
```bash
GET /backend/jobs/finished

Response: {"status": "finished", "jobs": ["uuid1", "uuid2", ...]}
```

---

## 🧪 Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format code
black .

# Lint
ruff check .

# Type check
mypy .
```

### Test Coverage

```bash
coverage run -m pytest
coverage report
```

---

## 📊 Dataset: `dane AG 2`

The repository includes benchmark knapsack problem instances:

```
dane AG 2/
├── large_scale/               # Large instances (knapPI_*)
├── large_scale-optimum/       # Known optimal values
├── low-dimensional/           # Smaller instances (f*)
└── low-dimensional-optimum/   # Known optimal values
```

**File format:** Each line contains `value weight`

---
[//]: <> ( ## Docker & Docker Compose)
[//]: <> ()
[//]: <> (Coming soon! The project will support:)
[//]: <> (- Multi-stage Docker build)
[//]: <> (- Docker Compose with API + PostgreSQL)
[//]: <> (- Automated CI/CD with Docker image publishing)


 Author: **Inez Małecka** – [maleckainez@gmail.com](mailto:maleckainez@gmail.com)

