"""Data schemas for the genetic algorithm API."""

from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    """Lifecycle states for a GA job."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"


class SelectionType(str, Enum):
    """Parent selection strategies."""

    ROULETTE = "roulette"
    TOURNAMENT = "tournament"
    LINEAR_RANK = "rank"


class CrossoverType(str, Enum):
    """Supported crossover operators."""

    ONE_POINT = "one"
    TWO_POINT = "two"


class LogLevel(str, Enum):
    """Allowed logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DataConfig(BaseModel):
    """Dataset source and constraints."""

    filename: str
    max_weight: int


class PopulationConfig(BaseModel):
    """Population size and iteration settings."""

    size: int
    generations: int
    stream_batch_size: int


class SelectionConfig(BaseModel):
    """Selection strategy parameters."""

    type: SelectionType
    selection_pressure: float


class GeneticOperatorsConfig(BaseModel):
    """Genetic operator probabilities and penalties."""

    crossover_type: CrossoverType
    crossover_probability: float
    mutation_probability: float
    penalty_multiplier: float


class ExperimentConfig(BaseModel):
    """Experiment metadata and logging level."""

    seed: int
    identifier: int
    log_level: LogLevel


class JobConfig(BaseModel):
    """Aggregate configuration for a GA run."""

    data: DataConfig
    population: PopulationConfig
    selection: SelectionConfig
    genetic_operators: GeneticOperatorsConfig
    experiment: ExperimentConfig


class JobStatusResponse(BaseModel):
    """Status payload returned by job endpoints."""

    job_id: str
    status: JobStatus
