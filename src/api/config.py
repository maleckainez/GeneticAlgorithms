class JobStatus(str, Enum):
    """Lifecycle states for a GA job."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"


class Job(BaseModel):
    job_id: str
    config: JobConfig  # Twój input
    status: JobStatus
    created_at: datetime
    updated_at: datetime
