"""Backend routes for genetic algorithm job handling."""

from fastapi import APIRouter

from src.api.schemas import JobConfig, JobStatus, JobStatusResponse
from src.api.services.runner import get_job_list, get_status, submit_job

router = APIRouter(prefix="/backend", tags=["backend"])


@router.post("/run/ga")
def run_job(config: JobConfig) -> JobStatusResponse:
    """Create a GA job and return its initial status.

    Args:
        config: ``JobConfig`` payload defining the GA run.
    """
    id = submit_job(config=config)
    status = get_status(id)
    return JobStatusResponse(job_id=id, status=status)


@router.get("/status/{job_id}")
def get_job_status(job_id: str) -> JobStatusResponse:
    """Fetch status of a GA job.

    Args:
        job_id: Identifier returned by ``run_job``.
    """
    status = get_status(job_id)
    return JobStatusResponse(job_id=job_id, status=status)


@router.get("/jobs/{status}")
def get_jobs(status: JobStatus) -> dict:
    """List job identifiers filtered by status.

    Args:
        status: Desired ``JobStatus`` filter.
    """
    jobs = get_job_list(status)
    return {"status": status, "jobs": jobs}
