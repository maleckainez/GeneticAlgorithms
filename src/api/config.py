"""Configuration models for genetic algorithm API jobs."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from src.ga_core.config import JobConfig


class JobStatus(str, Enum):
    """Lifecycle states for a GA job."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"


class Job(BaseModel):
    """Job metadata and state."""

    job_id: str
    config: JobConfig
    status: JobStatus
    created_at: datetime
    updated_at: datetime
