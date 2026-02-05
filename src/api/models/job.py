"""SQLAlchemy models for job tracking."""

from datetime import datetime
from typing import TYPE_CHECKING

import sqlalchemy as sql
from src.api.config import JobStatus
from src.api.db.database import Base

if TYPE_CHECKING:
    from sqlalchemy.sql.schema import Column


class Job(Base):
    """Database model for genetic algorithm job."""

    __tablename__ = "jobs"

    job_id: Column[str] = sql.Column(sql.String, primary_key=True)
    status: Column[JobStatus] = sql.Column(
        sql.Enum(JobStatus), default=JobStatus.PENDING
    )
    creation_date: Column[datetime] = sql.Column(sql.DateTime, default=datetime.utcnow)
    config_json: Column[str] = sql.Column(sql.String)
