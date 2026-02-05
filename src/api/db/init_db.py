"""Database initialization utilities."""

from src.api.db.database import Base, engine


def init_db() -> None:
    """Create all database tables from SQLAlchemy models."""
    Base.metadata.create_all(engine)
