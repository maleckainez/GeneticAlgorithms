"""Public storage API for genetic-algorithm experiments.

This package provides the main classes used to manage experiment files and
directories. It exposes the protocol describing required paths, a layout
implementation based on a root directory, and a helper class that groups
common storage operations needed during a run.

The default high-level entry point is ``ExperimentStorage``.
"""

from .experiment_storage import ExperimentStorage
from .layout import StorageLayout
from .naming_scheme import NAMING, DefaultFileNamingScheme, FileNamingScheme

__all__ = [
    "StorageLayout",
    "ExperimentStorage",
    "NAMING",
    "DefaultFileNamingScheme",
    "FileNamingScheme",
]
