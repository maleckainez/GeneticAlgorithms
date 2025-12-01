"""Helpers for reading experiment CSV outputs and metadata."""

import csv
from pathlib import Path
from typing import Any, Mapping

from pandas import DataFrame, read_csv

from src.ga_core.logging import LoggerType
from src.ga_core.output import CsvHandler


class ExperimentReader:
    """Read experiment results and metadata stored in CSV files."""

    def __init__(self, csv_path: Path) -> None:
        """Initialise a reader for a specific CSV file."""
        self._csv_path = csv_path

    @classmethod
    def from_handler(cls, handler: CsvHandler) -> "ExperimentReader":
        """Build a reader using a ``CsvHandler`` instance."""
        return cls(handler.filename)

    def load_results(self, comment: str = "#", logger: LoggerType = None) -> DataFrame:
        """Load numeric results, skipping metadata lines prefixed with ``#``.

        Args:
            comment: Lines starting with this marker are treated as metadata.
            logger: Optional logger for reporting errors or progress.

        Returns:
            DataFrame: Parsed results with comment lines removed.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
        """
        if not self._csv_path.exists():
            if logger is not None:
                logger.error("CSV file %s not found.", self._csv_path)
            raise FileNotFoundError(f"CSV file {self._csv_path} not found.")
        if logger is not None:
            logger.debug("Reading results from %s", self._csv_path)
        return read_csv(self._csv_path, comment=comment)

    def load_metadata(
        self, comment: str = "#", logger: LoggerType = None
    ) -> Mapping[str, Any]:
        """Load metadata rows formatted as ``#<key>,<value>``.

        Args:
            comment: Lines starting with this marker are treated as metadata.
            logger: Optional logger for reporting errors or progress.

        Returns:
            Mapping[str, Any]: Metadata key/value pairs parsed from the file.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
        """
        if not self._csv_path.exists():
            if logger is not None:
                logger.error("CSV file %s not found.", self._csv_path)
            raise FileNotFoundError(f"CSV file {self._csv_path} not found.")
        metadata: dict[str, Any] = {}
        if logger is not None:
            logger.debug("Reading metadata from %s", self._csv_path)
        with open(self._csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    break
                raw_key = row[0]

                if not raw_key.startswith(comment):
                    break
                key = raw_key[1:]
                value = row[1] if len(row) > 1 else None
                metadata[key] = value
        if logger is not None:
            logger.debug("Metadata read successfully from %s", self._csv_path)
        return metadata
