"""FullPlot public package interface.

FullPlot is a small engineering-data plotting package centered on HDF5 files,
reusable one-dimensional ``Trace`` objects, shared ``TimeAxis`` objects, and
rectangular-grid map generation. Most users import the package as ``fullplot``
or ``fplt`` and use the names re-exported here instead of importing from the
internal modules directly.
"""

from fullplot.cli import main
from fullplot.fullplot import (
    AmbiguousDatasetError,
    DatasetNotFoundError,
    FullPlotError,
    H5File,
    PlotDataError,
    TimeAxis,
    Trace,
    VALID_TRACE_ROLES,
    list,
    map,
    open,
    plot,
    read,
    show,
    time,
    trace,
    tree,
    values,
    write_traces,
)
from fullplot.maps import (
    Axis,
    FullPlotMapError,
    MapGenerationError,
    MapOutputError,
    generate_map,
)

__version__ = "0.1.1"

__all__ = [
    "__version__",
    "FullPlotError",
    "DatasetNotFoundError",
    "AmbiguousDatasetError",
    "PlotDataError",
    "TimeAxis",
    "Trace",
    "VALID_TRACE_ROLES",
    "H5File",
    "open",
    "tree",
    "list",
    "read",
    "values",
    "time",
    "trace",
    "plot",
    "map",
    "write_traces",
    "show",
    "Axis",
    "FullPlotMapError",
    "MapGenerationError",
    "MapOutputError",
    "generate_map",
    "main",
]
