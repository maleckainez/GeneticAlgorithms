"""Defines schemas for pydantic config validation."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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
    """Dataset source and weight constraint configuration.

    Captures the source filename and the per-instance maximum weight.
    """

    filename: str
    max_weight: int = Field(gt=0, description="Max weight must be greater than 0.")


class PopulationConfig(BaseModel):
    """Population size and iteration settings.

    Enforces an even population size, positive generation count, and
    batch size used for memmap streaming.
    """

    size: int = Field(
        ge=10, description="Population must count at least 10 individuals and be even."
    )
    generations: int = Field(
        gt=0, description="Number 0f generations should be greater than 0."
    )
    stream_batch_size: int = Field(
        default=500, gt=0, description="Batch size must be > 0"
    )

    @field_validator("size")
    @classmethod
    def size_must_be_even(cls, v: int) -> int:
        """Ensure population size is an even number."""
        if v % 2 != 0:
            raise ValueError("Population size must be even!")
        return v


class SelectionConfig(BaseModel):
    """Selection strategy parameters.

    Validates auxiliary fields required by the chosen selection type
    (selection_pressure for rank, tournament_size for tournament).
    """

    type: SelectionType
    selection_pressure: Optional[float]
    tournament_size: Optional[int]

    @model_validator(mode="after")
    def validate_rank_selection(self) -> SelectionConfig:
        """Guard linear rank configuration (selection_pressure in [1.0, 2.0])."""
        if self.type == SelectionType.LINEAR_RANK:
            if self.selection_pressure is None:
                raise ValueError("Selection pressure not specified !")
            if not 1.0 <= self.selection_pressure <= 2.0:
                raise ValueError("Selection pressure must be float in range [1.0, 2.0]")
        return self

    @model_validator(mode="after")
    def validate_tornament_size(self) -> SelectionConfig:
        """Guard tournament configuration (tournament_size in [2, 10])."""
        if self.type == SelectionType.TOURNAMENT:
            if self.tournament_size is None:
                raise ValueError("Tournament size not specified!")
            if not 2 <= self.tournament_size <= 10:
                raise ValueError("Tournament size must be integer in range [2, 10]")
        return self


class GeneticOperatorsConfig(BaseModel):
    """Genetic operator probabilities and penalties."""

    crossover_type: CrossoverType
    crossover_probability: float = Field(
        ge=0, le=1, description="Must be between 0 and 1"
    )
    mutation_probability: float = Field(
        ge=0, le=1, description="Must be between 0 and 1"
    )
    penalty_multiplier: float = Field(gt=0, description="Must be greater than 0")
    strict_weight_constraints: bool = Field(
        default=False,
        description=(
            "If True, any solution whose total weight exceeds max_weight, "
            "gets fitness set to 0 instead of receiving a penalty."
        ),
    )

    @model_validator(mode="after")
    def apply_weight_constraint(self) -> GeneticOperatorsConfig:
        """Set penalty to 0 when strict_weight_constraints is enabled."""
        if self.strict_weight_constraints:
            self.penalty_multiplier = 0.0
        return self


class ExperimentVals(BaseModel):
    """Experiment metadata and logging level."""

    seed: Optional[int] = None
    identifier: int
    log_level: LogLevel


class JobConfig(BaseModel):
    """Aggregate configuration for a GA run."""

    data: DataConfig
    population: PopulationConfig
    selection: SelectionConfig
    genetic_operators: GeneticOperatorsConfig
    experiment: ExperimentVals
