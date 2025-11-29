"""Validation schemas for GA input configuration."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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
    """Dataset source and weight constraint configuration."""

    data_filename: str
    max_weight: int = Field(gt=0, description="Max weight must be greater than 0.")


class PopulationConfig(BaseModel):
    """Population size and iteration settings."""

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
        """Ensure population size is an even number.

        Args:
            v: Proposed population size.

        Raises:
            ValueError: If the size is odd.

        Returns:
            int: The validated even size.
        """
        if v % 2 != 0:
            raise ValueError("Population size must be even!")
        return v


class SelectionConfig(BaseModel):
    """Selection strategy parameters and auxiliary fields."""

    type: SelectionType
    selection_pressure: Optional[float]
    tournament_size: Optional[int]

    @model_validator(mode="after")
    def validate_rank_selection(self) -> SelectionConfig:
        """Guard linear rank configuration.

        Raises:
            ValueError: If ``selection_pressure`` is missing or out of bounds.

        Returns:
            SelectionConfig: The validated instance.
        """
        if self.type == SelectionType.LINEAR_RANK:
            if self.selection_pressure is None:
                raise ValueError("Selection pressure not specified !")
            if not 1.0 <= self.selection_pressure <= 2.0:
                raise ValueError("Selection pressure must be float in range [1.0, 2.0]")
        return self

    @model_validator(mode="after")
    def validate_tornament_size(self) -> SelectionConfig:
        """Guard tournament configuration.

        Raises:
            ValueError: If ``tournament_size`` is missing or out of bounds.

        Returns:
            SelectionConfig: The validated instance.
        """
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
        """Set penalty to 0 when strict_weight_constraints is enabled.

        Returns:
            GeneticOperatorsConfig: The validated instance with updated penalty.
        """
        if self.strict_weight_constraints:
            self.penalty_multiplier = 0.0
        return self


class ExperimentVals(BaseModel):
    """Experiment metadata and logging level."""

    seed: Optional[int] = None
    identifier: Optional[str] = None
    log_level: LogLevel = LogLevel.INFO


class InputConfig(BaseModel):
    """Aggregate external input configuration for a GA run."""

    data: DataConfig
    population: PopulationConfig
    selection: SelectionConfig
    genetic_operators: GeneticOperatorsConfig
    experiment: ExperimentVals
