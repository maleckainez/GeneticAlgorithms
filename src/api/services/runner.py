"""Postgres job registry for genetic algorithm runs."""

import uuid
from fastapi import HTTPException
from src.api.config import JobConfig, JobStatus

from sqlalchemy.orm import Session
from src.api.db.database import SessionLocal
from src.api.models.job import Job

def submit_job(config: JobConfig) -> str:
    """Register a job and assign an identifier.

    Args:
        config: ``JobConfig`` payload describing the run.
    """
    db: Session = SessionLocal()
    job = Job(
        job_id=str(uuid.uuid4()),
        status=JobStatus.PENDING,
        config_json=config.model_dump_json(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    db.close()

    return job.job_id


def get_status(job_id: str) -> JobStatus:
    """Return job status or raise 404 when unknown.

    Args:
        job_id: Identifier returned by ``submit_job``.
    """
    db: Session = SessionLocal()
    job = db.query(Job).filter(Job.job_id == job_id).first()
    db.close()
    if not job:
        raise HTTPException(status_code=404, detail="This job does not exist!")
    return job.status


def get_job_list(status: JobStatus) -> list:
    """Return job identifiers that match the given status.

    Args:
        status: Desired ``JobStatus`` filter.
    """
    db: Session = SessionLocal()
    job_list = db.query(Job).filter(Job.status == status).all()
    if not job_list:
        db.close()
        return []
    job_ids = [job.job_id for job in job_list]
    db.close()
    return job_ids
