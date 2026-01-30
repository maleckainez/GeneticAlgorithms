"""File naming conventions and resolver for GA experiment files.

This module centralizes all file naming patterns to maintain consistency
across the library and allow easy customization through protocols.
"""

from typing import Protocol


class FileNamingScheme(Protocol):
    """Protocol for customizable file naming conventions."""

    def population_file(self, job_id: str) -> str:
        """Return population memmap filename."""

    def children_file(self, job_id: str) -> str:
        """Return children memmap filename."""

    def population_config_file(self, job_id: str) -> str:
        """Return population config JSON filename."""

    def log_file(self, job_id: str) -> str:
        """Return log filename."""

    def csv_file(self, exp_name: str) -> str:
        """Return CSV output filename."""


class DefaultFileNamingScheme:
    """Default naming convention for GA experiment files."""

    def population_file(self, job_id: str) -> str:
        """Population memmap: <job_id>.dat."""
        return f"{job_id}.dat"

    def children_file(self, job_id: str) -> str:
        """Children memmap: <job_id>_2.dat."""
        return f"{job_id}_2.dat"

    def population_config_file(self, job_id: str) -> str:
        """Config JSON: <job_id>.json."""
        return f"{job_id}.json"

    def log_file(self, job_id: str) -> str:
        """Log file: runtime_experiment_<job_id>.log."""
        return f"runtime_experiment_{job_id}.log"

    def csv_file(self, exp_name: str) -> str:
        """CSV output: <exp_name>.csv."""
        return f"{exp_name}.csv"


# Default global instance
NAMING = DefaultFileNamingScheme()
