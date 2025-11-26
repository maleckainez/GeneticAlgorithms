"""Expose FastAPI app with backend router."""

from fastapi import FastAPI

from src.api.routers import backend

app = FastAPI()

app.include_router(backend.router)
