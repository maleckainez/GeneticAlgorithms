"""Data schemas for the genetic algorithm API."""

from pydantic import BaseModel

from src.config.schemas import JobStatus


class JobStatusResponse(BaseModel):
    """Status payload returned by job endpoints."""

    job_id: str
    status: JobStatus
