"""Simple schemas for API job handling - MVP version."""

from enum import Enum

from src.ga_core.config import InputConfig

# Re-export InputConfig as JobConfig for API
JobConfig = InputConfig


class JobStatus(str, Enum):
    """Job lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
