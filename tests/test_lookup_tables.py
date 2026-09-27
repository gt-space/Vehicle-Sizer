import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from FluidTables.LookupTables import LookupTable, LookupTables


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
    lookups = LookupTables(root / "FluidTables" / "sizer_lookups.h5")
    engine = lookups["engine_lookup"]
    coordinates = {name: axis[len(axis) // 2] for name, axis in engine.axes.items()}

    engine.get("characteristic_velocity", **coordinates)
    coordinates["nfz"] = 0.5
    with pytest.raises(ValueError, match="requires one of"):
        engine.get("characteristic_velocity", **coordinates)


def test_hermite_values_and_derivatives_share_a_continuous_interpolant():
    p, t = np.array([1., 2.5, 4.]), np.array([10., 12., 15.])
    x, y = np.meshgrid(p, t, indexing='ij')
    outputs = dict(density=x*x*y*y + 10., density_dP=2*x*y*y,
                   density_dT=2*y*x*x, density_dPdT=4*x*y)
    table = LookupTable('pt', ('pressure', 'temperature'), dict(pressure=p, temperature=t),
                        outputs, {}, {}, {'hermite_outputs': {'density':
                        ['density_dP', 'density_dT', 'density_dPdT']}}, {},
                        {'success': np.ones((3, 3), dtype=bool)})
    for pressure, temperature in [(1., 10.), (4., 15.), (1.7, 11.2), (2.5, 12.), (3.1, 13.)]:
        at = dict(pressure=pressure, temperature=temperature)
        result = table.evaluate(**at)
        assert result['density'] == pytest.approx(pressure**2 * temperature**2 + 10.)
        assert result['density_dP'] == pytest.approx(2*pressure*temperature**2)
        assert table.derivative(['density'], wrt='temperature', **at)['density'] == pytest.approx(2*temperature*pressure**2)
    for axis, knot in [('pressure', 2.5), ('temperature', 12.)]:
        a = dict(pressure=2.5, temperature=12.)
        b = dict(a)
        a[axis], b[axis] = knot-1e-8, knot+1e-8
        for name in ['density', 'density_dP', 'density_dT']:
            assert table.get(name, **a) == pytest.approx(table.get(name, **b), rel=1e-7)
    table.status['success'][2, 2] = False
    with pytest.raises(ValueError, match='unsuccessful'):
        table.get('density', pressure=3., temperature=14.)
    # A valid neighboring cell supplies the value and slopes at its boundary.
    assert table.get('density', pressure=2.5, temperature=14.) == pytest.approx(2.5**2 * 14.**2 + 10.)
    with pytest.raises(ValueError, match='outside'):
        table.derivative(['density'], wrt='pressure', pressure=0., temperature=12.)


def test_generated_pt_table_stores_coolprop_slopes_and_runtime_uses_them(tmp_path):
    from CoolProp import AbstractState
    from CoolProp.CoolProp import PT_INPUTS, iDmass, iUmass, iP, iT
    from FluidTables.pt_lookup import PTSpecification, generate_pt_table
    from FluidTables.PropertyModels import TablePureFluidPropertySource
    path = tmp_path / 'properties.h5'
    generate_pt_table(PTSpecification('Nitrogen', 'nitrogen_pt', (1e6, 5e6), (280., 320.), 8, 9, 'gas'), path)
    source = TablePureFluidPropertySource(path, {'Nitrogen': 'nitrogen_pt'})
    table = source._table('Nitrogen')
    p, t = table.axes['pressure'][3], table.axes['temperature'][4]
    reference = AbstractState('HEOS', 'Nitrogen')
    reference.update(PT_INPUTS, p, t)
    for name, key in [('density', iDmass), ('internal_energy', iUmass)]:
        assert table.outputs[name+'_dP'][3, 4] == pytest.approx(reference.first_partial_deriv(key, iP, iT), rel=1e-10)
        assert table.outputs[name+'_dT'][3, 4] == pytest.approx(reference.first_partial_deriv(key, iT, iP), rel=1e-10)
        assert table.outputs[name+'_dPdT'][3, 4] == pytest.approx(reference.second_partial_deriv(key, iP, iT, iT, iP), rel=1e-10)
    p, t = 2.3e6, 301.7
    reference.update(PT_INPUTS, p, t)
    state, slopes = source.state_pt('Nitrogen', p, t), source.derivatives_pt('Nitrogen', p, t)
    assert state.rho == pytest.approx(reference.rhomass(), rel=1e-5)
    assert state.u == pytest.approx(reference.umass(), rel=1e-5)
    assert state.h == pytest.approx(state.u + p/state.rho, rel=1e-14)
    assert state.beta == pytest.approx(-slopes['drho_dT']/state.rho)
    for axis, step, suffix in [('pressure', 10., 'P'), ('temperature', .001, 'T')]:
        at = dict(pressure=p, temperature=t)
        low, high = dict(at), dict(at)
        low[axis] -= step
        high[axis] += step
        a, b = source.state_pt('Nitrogen', **low), source.state_pt('Nitrogen', **high)
        assert slopes['drho_d'+suffix] == pytest.approx((b.rho-a.rho)/(2*step), rel=1e-6)
        assert slopes['du_d'+suffix] == pytest.approx((b.u-a.u)/(2*step), rel=1e-6)
