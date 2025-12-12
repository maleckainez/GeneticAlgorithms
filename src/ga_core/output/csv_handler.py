"""CSV file helper used by output utilities.

This module only opens and closes files. It does not create missing directories,
so the caller must ensure ``output_path`` already exists.
"""

import csv
from pathlib import Path
from types import TracebackType
from typing import Any, Optional, TextIO

from src.ga_core.storage.naming_scheme import NAMING


class CsvHandler:
    """Open, write, and close experiment CSV outputs."""

    def __init__(self, output_path: Path, exp_filename: str) -> None:
        """Construct a handler for a single experiment CSV file.

        Args:
            output_path: Directory where the CSV file will be written.
            exp_filename: Base filename without extension.
        """
        self._filename = output_path / NAMING.csv_file(exp_filename)
        self._file: Optional[TextIO] = None
        self._writer: Optional[Any] = None

    def open(self) -> None:
        """Open the CSV file and prepare the writer."""
        if self._file is None:
            self._file = open(self._filename, "w", newline="")
            self._writer = csv.writer(self._file)

    def close(self) -> None:
        """Close the CSV file and reset internal state."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    @property
    def writer(self) -> Any:
        """Return the underlying csv.writer object.

        Raises:
            RuntimeError: If ``open`` has not been called yet.
        """
        if self._writer is None:
            raise RuntimeError("CSV file is not opened!")
        return self._writer

    def __enter__(self) -> "CsvHandler":
        """Open the handler as a context manager."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None:
        """Close the handler when exiting a context block."""
        self.close()

    @property
    def filename(self) -> Path:
        """Return the full path to the managed CSV file."""
        return self._filename
