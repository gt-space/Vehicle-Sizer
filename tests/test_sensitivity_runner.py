from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from diagnostics.constraints import vehicle_limit_records
from sensitivity import apply_case, make_cases, run_sweep
from sensitivity.parameters import sweep_values
from diagnostics.errors import EvaluationFailure, ModelDomainExceeded


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


def simulate_with_failed_middle_case(candidate, **kwargs):
    value = candidate['engine']['cstar_efficiency']
    if value == 0.9:
        raise EvaluationFailure('runtime', candidate, FakeResult(apogee=999),
                                warnings=[dict(code='example_warning', message='test caution', count=1)]) from ModelDomainExceeded('AoA exceeded 15 deg')
    return FakeResult(apogee=value ** 2, feasible=False)


@pytest.mark.parametrize('workers', [1, 2])
def test_sweeps_continue_after_failure_and_never_export_partial_outputs(tmp_path, workers):
    import csv
    base = {'engine': {'cstar_efficiency': 0.8}}
    spec = {'sweep': {'workers': workers, 'parameters': {
        'engine.cstar_efficiency': {'values': [0.8, 0.9, 1.0]}}},
        'outputs': {'record_history': False, 'directory': str(tmp_path)}}
    result = run_sweep(base, spec, simulate_fn=simulate_with_failed_middle_case, reuse_models=False)
    assert len(result.summary) == 3
    failed = result.by_id['case_00001']
    assert failed['status'] == 'error'
    assert failed.get('apogee') in ('', None)
    assert 'ModelDomainExceeded' in failed['error']
    assert result.by_id['case_00002']['status'] == 'completed'
    assert len(result.scalar_points('engine.cstar_efficiency', 'apogee')) == 2
    error = list(csv.DictReader((tmp_path / 'errors.csv').open()))[0]
    assert error['kind'] == 'table_domain_exceeded'
    assert error['exception'] == 'ModelDomainExceeded'
    warning = list(csv.DictReader((tmp_path / 'warnings.csv').open()))[0]
    assert warning['code'] == 'example_warning' and warning['case_id'] == 'case_00001'


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


def test_linked_mass_sweep_updates_both_prop_tanks_only():
    base = {'vehicle': {'sections': [
        {'tank_id': 'ox_tank', 'masses': {'extra': 0.0}},
        {'tank_id': 'fuel_tank', 'masses': {'extra': 0.0}},
        {'tank_id': 'press_tank', 'masses': {'shell': 6.89}},
    ]}}
    original = deepcopy(base)
    spec = {'sweep': {'parameters': {'extra_mass': {
        'path': 'vehicle.sections[0].masses.extra',
        'linked_paths': ['vehicle.sections[1].masses.extra'],
        'values': [0.0, 10.0, 20.0],
    }}}}
    parameters, cases = make_cases(base, spec)
    assert len(cases) == 3
    changed = apply_case(base, cases[-1], parameters)
    assert [s['masses'] for s in changed['vehicle']['sections']] == [
        {'extra': 20.0}, {'extra': 20.0}, {'shell': 6.89}]
    assert base == original


def test_linked_paths_reject_overlap_and_invalid_values():
    base = {'engine': {'cstar_efficiency': 0.9, 'cf_efficiency': 0.95}}
    spec = {'sweep': {'parameters': {'efficiency': {
        'path': 'engine.cstar_efficiency', 'linked_paths': ['engine.cf_efficiency'],
        'values': [0.9, 1.0],
    }}}}
    parameters, cases = make_cases(base, spec)
    assert len(cases) == 3  # 0.9 is not a duplicate: linked Cf differs at baseline.
    assert apply_case(base, cases[1], parameters)['engine']['cf_efficiency'] == 0.9
    spec['sweep']['parameters']['second'] = {'path': 'engine.cf_efficiency', 'values': [0.8]}
    with pytest.raises(ValueError, match='same configuration path'):
        make_cases(base, spec)


@pytest.mark.parametrize('workers', [1, 2])
def test_linked_offsets_preserve_distinct_baselines_and_saved_inputs(tmp_path, workers):
    base = {'engine': {'cstar_efficiency': 0.9, 'cf_efficiency': 0.85},
            'environment': {'wind': {'enabled': True}}}
    original = deepcopy(base)
    spec = {'sweep': {'workers': workers, 'ignore_wind': True, 'parameters': {'delta': {
        'path': 'engine.cstar_efficiency', 'linked_paths': ['engine.cf_efficiency'],
        'offset': True, 'values': [-0.05, 0, 0.05],
    }}}, 'outputs': {'record_history': False, 'directory': str(tmp_path)}}
    parameters, cases = make_cases(base, spec)
    assert len(cases) == 3
    changed = apply_case(base, cases[-1], parameters)
    assert changed['engine']['cstar_efficiency'] == pytest.approx(0.95)
    assert changed['engine']['cf_efficiency'] == pytest.approx(0.9)
    result = run_sweep(base, spec, simulate_fn=fake_simulate, reuse_models=False)
    assert base == original
    assert result.by_id['baseline']['delta'] == 0
    row = result.by_id[cases[-1].case_id]
    assert row['delta'] == 0.05
    assert row['inputs.engine.cstar_efficiency'] == pytest.approx(0.95)
    assert row['inputs.engine.cf_efficiency'] == pytest.approx(0.9)
    assert row['apogee'] == pytest.approx(0.95 ** 2)


def test_offsets_validate_actual_types_on_linked_paths():
    base = {'engine': {'cstar_efficiency': 0.8}, 'fin_can': {'fin_count': 4}}
    spec = {'sweep': {'parameters': {'delta': {
        'path': 'engine.cstar_efficiency', 'linked_paths': ['fin_can.fin_count'],
        'offset': True, 'values': [0.1],
    }}}}
    with pytest.raises(ValueError, match='whole-number'):
        make_cases(base, spec)


def test_physical_range_endpoints_are_left_to_simulate_per_case():
    base = {'engine': {'cstar_efficiency': 0.9}}
    spec = {'sweep': {'parameters': {'engine.cstar_efficiency': {'values': [-0.1, 0.9, 1.0]}}}}
    parameters, cases = make_cases(base, spec)
    assert len(cases) == 3
    assert apply_case(base, cases[1], parameters)['engine']['cstar_efficiency'] == -0.1


def test_vehicle_limits_report_margins_without_rejecting():
    cfg = {'constraints': {'max_length_to_diameter': 25.0}, 'vehicle': {'OMLD': 0.25}}
    assert vehicle_limit_records(cfg, SimpleNamespace(length=7.5))['max_length_to_diameter'].margin == -5.0


def test_design_aoa_limit_does_not_control_model_domain():
    from dataclasses import replace
    from diagnostics.errors import ModelDomainExceeded
    from simulation_types import KinematicsState
    from test_simulation_api import make_flight
    flight = make_flight()
    flight.cfg['constraints'] = {'max_aoa_deg': 5.0}
    start = KinematicsState(t=0, dt=0.1, x=0, h=10, vx=0, vz=10,
                            theta=np.pi / 2, q=0, alpha=np.deg2rad(10), m=10, Iyy=1)
    end = replace(start, t=0.1, h=11)
    assert flight.check_commit(start, end) is None
    from simulation_types import SimResult
    result = SimResult()
    atmosphere = flight._atmosphere(end)
    flight._record_extrema(result, atmosphere, flight.trial_aero(end, atmosphere, True), end)
    assert result.max_aoa_deg == pytest.approx(10)
    assert result.constraint_records == {}
    with pytest.raises(ModelDomainExceeded, match='AoA'):
        flight.check_commit(start, replace(end, alpha=np.deg2rad(16)))
