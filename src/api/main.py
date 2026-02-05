"""Expose FastAPI app with backend router."""

from fastapi import FastAPI

from src.api.db.init_db import init_db
from src.api.routers import backend

app = FastAPI(
    title="Genetic Algorithms API",
    description="API for running genetic algorithm experiments on knapsack problems",
    version="0.1.0",
)

app.include_router(backend.router)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database tables."""
    init_db()


@app.get("/")
def root() -> dict:
    """Root endpoint - API info."""
    return {
        "name": "Genetic Algorithms API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}
