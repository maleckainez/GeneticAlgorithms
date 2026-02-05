import sqlalchemy as sql
from datetime import datetime
from src.api.config import JobStatus
from src.api.db.database import Base

class Job(Base):
    __tablename__ = "jobs"

    job_id = sql.Column(sql.String, primary_key=True)
    status = sql.Column(sql.Enum(JobStatus), default=JobStatus.PENDING)
    creation_date = sql.Column(sql.DateTime, default=datetime.utcnow)
    config_json = sql.Column(sql.String)