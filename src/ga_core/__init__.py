"""Public API for genetic-algorithm experiments.

This module exposes the main high-level entry points used to configure and
run experiments.
"""

from .storage import ExperimentStorage, PathResolver, StorageLayout
from .storage import ExperimentStorage as Storage

__all__ = [
    "Storage",
    "ExperimentStorage",
    "StorageLayout",
    "PathResolver",
]
