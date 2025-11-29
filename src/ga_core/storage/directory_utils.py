"""Helpers for creating and cleaning experiment directories."""

from .layout import StorageLayout


def ensure_layout_paths(layout: StorageLayout) -> None:
    """Create all directories defined by the layout if they do not exist.

    Args:
        layout: Layout describing ``temp``, ``output``, ``logs``, and ``plots``
            directories. Missing directories are created recursively.
    """
    for path in (layout.temp, layout.output, layout.logs, layout.plots):
        path.mkdir(parents=True, exist_ok=True)


def cleanup_temp(layout: StorageLayout) -> None:
    """Remove the temporary directory and its contents, if it exists.

    Args:
        layout: Layout providing the temporary directory path. If the directory
            does not exist, no action is taken.
    """
    temp = layout.temp
    if temp.exists():
        import shutil

        shutil.rmtree(temp)
