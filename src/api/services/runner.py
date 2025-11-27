"""In-memory job registry for genetic algorithm runs."""

import uuid

from fastapi import HTTPException
from src.api.schemas import JobStatus
from src.config.schemas import JobConfig

jobs = {}


def submit_job(config: JobConfig) -> str:
    """Register a job and assign an identifier.

    Args:
        config: ``JobConfig`` payload describing the run.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "config": config}
    return job_id


def get_status(job_id: str) -> JobStatus:
    """Return job status or raise 404 when unknown.

    Args:
        job_id: Identifier returned by ``submit_job``.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="This id does not exist!")
    return JobStatus(jobs[job_id]["status"])


def get_job_list(status: JobStatus) -> list:
    """Return job identifiers that match the given status.

    Args:
        status: Desired ``JobStatus`` filter.
    """
    return [job_id for job_id in jobs if jobs[job_id]["status"] == status]
