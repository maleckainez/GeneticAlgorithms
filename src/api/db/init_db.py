from src.api.db.database import engine, Base
from src.api.models.job import Job

def init_db() -> None:
    Base.metadata.create_all(engine)