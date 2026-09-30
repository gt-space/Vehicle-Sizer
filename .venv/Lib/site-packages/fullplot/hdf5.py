"""Small HDF5 path and naming helpers used by FullPlot."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import h5py


HDF5_EXTENSIONS = {".h5", ".hdf5"}


def hdf5_path(filename: str | Path) -> Path:
    """Return a normalized HDF5 path for writing.

    A missing suffix is treated as ``.h5`` so calls such as
    ``write_traces("run", traces)`` create ``run.h5``. Existing ``.h5`` and
    ``.hdf5`` suffixes are preserved. Any other suffix raises ``ValueError`` to
    avoid accidentally writing HDF5 content to a misleading filename. Parent
    directories are created when needed.
    """
    path = Path(filename)

    if path.suffix == "":
        path = path.with_suffix(".h5")
    elif path.suffix.lower() not in HDF5_EXTENSIONS:
        raise ValueError("Use an HDF5 filename with no extension, .h5, or .hdf5.")

    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)

    return path


def hdf5_filename(filename: str | Path) -> str:
    """Return a normalized HDF5 filename string for writing.

    This is a convenience wrapper around :func:`hdf5_path` for APIs that pass
    filenames directly to ``h5py.File``.
    """
    return str(hdf5_path(filename))


def safe_group_name(name: Any) -> str:
    """Return a compact HDF5-safe group or dataset name.

    Slashes, null bytes, and whitespace are replaced so trace names such as
    ``"PCMC Redline"`` can safely become HDF5 group names. The transformation is
    deliberately conservative and human-readable; it is not a general-purpose
    slugification routine.
    """
    text = str(name).strip()
    text = re.sub(r"[\\/\x00]", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text or "unnamed"


def dataset_names(group: h5py.Group, excluded: set[str] | None = None) -> list[str]:
    """Return direct child dataset names from an HDF5 group.

    Parameters
    ----------
    group:
        Open ``h5py.Group`` to inspect.
    excluded:
        Optional names to omit from the result. This is mainly used by map
        generation when status datasets should not be considered output maps.
    """
    excluded = set() if excluded is None else set(excluded)
    return [
        name
        for name, item in group.items()
        if name not in excluded and isinstance(item, h5py.Dataset)
    ]
