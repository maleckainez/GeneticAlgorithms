"""Database connection and session management."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


def get_database_url() -> str:
    """Build database URL from environment variables.

    Raises:
        ValueError: If required environment variables are not set.
    """
    # Allow full DATABASE_URL override
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url

    # Otherwise require individual components
    db_host = os.getenv("POSTGRES_HOST")
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_name = os.getenv("POSTGRES_DB")
    db_port = os.getenv("POSTGRES_PORT", "5432")

    missing = []
    if not db_host:
        missing.append("POSTGRES_HOST")
    if not db_user:
        missing.append("POSTGRES_USER")
    if not db_password:
        missing.append("POSTGRES_PASSWORD")
    if not db_name:
        missing.append("POSTGRES_DB")

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Set DATABASE_URL or individual POSTGRES_* variables."
        )

    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


DB_URL = get_database_url()
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()
