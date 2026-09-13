import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from FluidProperties.LookupTables import LookupTables


def make_lookup_file(path):
    with h5py.File(path, "w") as file:
        file.attrs["map_schema"] = "rectangular-grid-map-v1"
        group = file.create_group("sample")
        group.attrs["axis_order"] = json.dumps(["pressure", "mode"])
        group.attrs["output_names"] = json.dumps(["density", "invalid"])
        group.attrs["metadata"] = json.dumps({"fluid": "sample"})
        group.attrs["constants"] = json.dumps({"reference": 1.0})

        axes = group.create_group("axes")
        pressure = axes.create_dataset("pressure", data=[1.0, 3.0])
        pressure.attrs["units"] = "Pa"
        mode = axes.create_dataset("mode", data=[0.0, 1.0])
        mode.attrs["units"] = ""

        outputs = group.create_group("outputs")
        density = outputs.create_dataset(
            "density", data=[[10.0, 20.0], [30.0, 40.0]]
        )
        density.attrs["units"] = "kg/m^3"
        outputs.create_dataset("invalid", data=[[1.0, 2.0], [3.0, np.nan]])

        status = group.create_group("status")
        status.create_dataset("success", data=np.ones((2, 2), dtype=bool))


def test_loads_metadata_raw_arrays_and_interpolates(tmp_path):
    path = tmp_path / "lookup.h5"
    make_lookup_file(path)
    lookups = LookupTables(path)

    assert lookups.names == ("sample",)
    table = lookups["sample"]
    assert table.axis_order == ("pressure", "mode")
    assert table.axis_units["pressure"] == "Pa"
    assert table.output_units["density"] == "kg/m^3"
    assert table.metadata["fluid"] == "sample"
    assert table.constants["reference"] == 1.0
    assert np.array_equal(table.axes["pressure"], [1.0, 3.0])
    assert table.outputs["density"].shape == (2, 2)
    assert table.status["success"].all()
    assert lookups.get("sample", "density", pressure=2.0, mode=0.5) == pytest.approx(
        25.0
    )


def test_evaluate_returns_selected_or_all_outputs(tmp_path):
    path = tmp_path / "lookup.h5"
    make_lookup_file(path)
    table = LookupTables(path)["sample"]

    assert table.evaluate(["density"], pressure=1.0, mode=0.0) == {
        "density": 10.0
    }
    assert set(table.evaluate(pressure=1.0, mode=0.0)) == {"density", "invalid"}


def test_repeated_query_uses_interpolation_cache(tmp_path):
    path = tmp_path / "lookup.h5"
    make_lookup_file(path)
    table = LookupTables(path)["sample"]
    coordinates = {"pressure": 2.0, "mode": 0.5}

    table.get("density", **coordinates)
    hits = table._interpolate.cache_info().hits
    table.get("density", **coordinates)

    assert table._interpolate.cache_info().hits == hits + 1


def test_invalid_queries_raise_and_warn_without_extrapolating(tmp_path, capsys):
    path = tmp_path / "lookup.h5"
    make_lookup_file(path)
    table = LookupTables(path)["sample"]

    with pytest.raises(ValueError, match="missing"):
        table.get("density", pressure=1.0)
    with pytest.raises(ValueError, match="outside"):
        table.get("density", pressure=4.0, mode=0.0)
    assert "WARNING: rejected query outside table" in capsys.readouterr().err
    with pytest.raises(KeyError, match="no output"):
        table.get("temperature", pressure=1.0, mode=0.0)
    with pytest.raises(ValueError, match="non-finite"):
        table.get("invalid", pressure=3.0, mode=1.0)


def test_engine_nfz_is_discrete():
    root = Path(__file__).resolve().parents[1]
    lookups = LookupTables(root / "FluidProperties" / "sizer_lookups.h5")
    engine = lookups["engine_lookup"]
    coordinates = {name: axis[len(axis) // 2] for name, axis in engine.axes.items()}

    engine.get("characteristic_velocity", **coordinates)
    coordinates["nfz"] = 0.5
    with pytest.raises(ValueError, match="requires one of"):
        engine.get("characteristic_velocity", **coordinates)
