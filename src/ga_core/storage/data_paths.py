"""Locate dataset files bundled with this project."""

from pathlib import Path

BASE_DATA = Path("dane AG 2")

SMALL_SCALE_DIR = BASE_DATA / "low-dimensional"
SMALL_SCALE_OPTIMUM_DIR = BASE_DATA / "low-dimensional-optimum"
LARGE_SCALE_DIR = BASE_DATA / "large_scale"
LARGE_SCALE_OPTIMUM_DIR = BASE_DATA / "large_scale-optimum"


def resolve_data_dict_path(data_file_name: str) -> Path:
    """Return the absolute path to a dataset file based on its name prefix.

    Args:
        data_file_name: File stem of the instance, e.g. ``knapPI_1_100_1000_1``
            or ``f6_l-d_kp_10_60``. The prefix determines which directory is
            used.

    Raises:
        ValueError: If the name does not start with ``knap`` or ``f``.

    Returns:
        Path: Absolute path to the requested data file under the appropriate
        directory.
    """
    if data_file_name.startswith("knap"):
        return LARGE_SCALE_DIR / data_file_name
    if data_file_name.startswith("f"):
        return SMALL_SCALE_DIR / data_file_name
    raise ValueError("Invalid data file name prefix (expected 'knap' or 'f')")


def resolve_optimum_file(data_file_name: str) -> Path:
    """Return the file containing optimum solutions for the dataset.

    Args:
        data_file_name: File stem of the instance, e.g. ``knapPI_1_100_1000_1``
            or ``f6_l-d_kp_10_60``. The prefix determines in which directory is
            optimum file located.

    Raises:
        ValueError: If the name does not start with ``knap`` or ``f``.

    Returns:
        Path: File containing optimum solutions for the instance family.
    """
    if data_file_name.startswith("knap"):
        return LARGE_SCALE_OPTIMUM_DIR / data_file_name
    if data_file_name.startswith("f"):
        return SMALL_SCALE_OPTIMUM_DIR / data_file_name
    raise ValueError("Invalid data file name prefix (expected 'knap' or 'f')")
