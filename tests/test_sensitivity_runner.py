from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from diagnostics.constraints import DesignInfeasible, vehicle_limit_margins
from sensitivity import apply_case, make_cases, run_sweep
from sensitivity.parameters import sweep_values


@dataclass
class FakeResult:
    apogee: float
    completed: bool = True
    feasible: bool = True
    termination: str = 'apogee'
    history: object = None


def fake_simulate(candidate, **kwargs):
    assert candidate['environment']['wind']['enabled'] is False
    pressure = candidate['engine']['cstar_efficiency']
    if pressure == 0.7:
        raise RuntimeError('synthetic solver failure')
    candidate['engine']['cstar_efficiency'] = 0  # Must not leak into other cases.
    return FakeResult(apogee=pressure ** 2)


@pytest.mark.parametrize('workers', [1, 2])
def test_serial_and_parallel_sweeps_round_trip_without_mutating_base(tmp_path, workers):
    base = {'engine': {'cstar_efficiency': 0.9},
            'environment': {'wind': {'enabled': True}}}
    original = deepcopy(base)
    # No mode key: exercise defaulting through execution and results reload.
    spec = {'sweep': {'workers': workers, 'ignore_wind': True,
                      'parameters': {'engine.cstar_efficiency': {'values': [0.7, 0.8, 0.9, 1.0]}}},
            'outputs': {'record_history': False, 'directory': str(tmp_path)}}
    result = run_sweep(base, spec, simulate_fn=fake_simulate, reuse_models=False)
    assert base == original
    assert list(result.by_id) == ['baseline', 'case_00001', 'case_00002', 'case_00003']
    assert result.by_id['case_00001']['status'] == 'error'
    assert 'synthetic solver failure' in result.by_id['case_00001']['error']
    assert result.by_id['baseline']['apogee'] == pytest.approx(0.81)
    assert [p[0] for p in result.scalar_points('engine.cstar_efficiency', 'apogee')] == [0.8, 0.9, 1.0]
    x, derivative = result.scalar_derivative('apogee', 'engine.cstar_efficiency')
    np.testing.assert_allclose(derivative, 2 * x)
    for name in ('summary.csv', 'cases.csv', 'errors.csv', 'baseline_config.yaml',
                 'sweep_config.yaml', 'field_catalog.csv', 'histories.h5'):
        assert (tmp_path / name).exists()


def test_full_grid_and_indexed_overrides_are_independent():
    base = {'engine': {'cstar_efficiency': 0.9},
            'vehicle': {'sections': [{'masses': {'cube': 4.0}}]}}
    original = deepcopy(base)
    spec = {'sweep': {'mode': 'full_grid', 'parameters': {
        'engine.cstar_efficiency': {'values': [0.8, 1.0]},
        'vehicle.sections[0].masses.cube': {'values': [3.0, 5.0]},
    }}}
    parameters, cases = make_cases(base, spec)
    assert len(cases) == 5
    changed = apply_case(base, cases[1], parameters)
    assert changed['engine']['cstar_efficiency'] == 0.8
    assert changed['vehicle']['sections'][0]['masses']['cube'] == 3.0
    assert base == original
    spec['sweep']['max_cases'] = 4
    with pytest.raises(ValueError, match='max_cases'):
        make_cases(base, spec)


def test_decimal_ranges_include_endpoint():
    assert sweep_values({'start': 0.06, 'stop': 0.08, 'step': 0.005}, 'volume') == (
        0.06, 0.065, 0.07, 0.075, 0.08)


def test_feasibility_bypass_preserves_static_margin():
    cfg = {'constraints': {'max_length_to_diameter': 25.0}, 'vehicle': {'OMLD': 0.25}}
    vehicle = SimpleNamespace(length=7.5)
    with pytest.raises(DesignInfeasible):
        vehicle_limit_margins(cfg, vehicle)
    assert vehicle_limit_margins(cfg, vehicle, reject_infeasible=False) == {
        'max_length_to_diameter': -5.0}


def test_aoa_bypass_keeps_normal_checks_by_default():
    from dataclasses import replace
    from diagnostics.constraints import OperatingInfeasible
    from simulation_types import KinematicsState
    from test_simulation_api import make_flight

    flight = make_flight()
    flight.cfg['constraints'] = {'max_aoa_deg': 5.0}
    start = KinematicsState(t=0, dt=0.1, x=0, h=10, vx=0, vz=10,
                            theta=np.pi / 2, q=0, alpha=np.deg2rad(10), m=10, Iyy=1)
    end = replace(start, t=0.1, h=11)
    with pytest.raises(OperatingInfeasible):
        flight.check_commit(start, end)
    flight.ignore_feasibility = True
    assert flight.check_commit(start, end) is None
