from copy import deepcopy

import numpy as np
import pytest

from Fluids.design import prepare_design_inputs
from Configs.loader import load_config
from Fluids.helpers.battery import size_battery
from Fluids.PropSystem import PropSystem
from Vehicle.Vehicle import Vehicle
from simulation import simulate
from FluidTables.PropertyModels import property_sources
from reporting.history import capture_history
from reporting.run_report import build_run_tables


@pytest.fixture(scope='module')
def design():
    cfg = load_config('Configs/flight_pump_fed_regulator_epump.yaml')
    # Exercise assembly with a design inside the 15 kW pump limits at 46% efficiency.
    cfg['prop_system']['thrust_target'] = 10000.
    pure, combustion = property_sources(cfg)
    template, states, wall_pressures = prepare_design_inputs(cfg)
    vehicle = Vehicle(cfg, pure, initial_states=states, wall_pressures=wall_pressures)
    propulsion = PropSystem(cfg, vehicle.tanks, fluid_properties=pure,
                            combustion_properties=combustion, initial_states=states, template=template)
    yield cfg, pure, combustion, propulsion
    propulsion.close()


def build(design, cfg=None):
    original, pure, _, propulsion = design
    cfg = deepcopy(original if cfg is None else cfg)
    template, states, wall_pressures = prepare_design_inputs(cfg)
    vehicle = Vehicle(cfg, pure, initial_states=states, wall_pressures=wall_pressures)
    vehicle.build(propulsion)
    return vehicle


def test_power_sizing_and_pump_sum(design):
    _, _, _, propulsion = design
    sized = size_battery({'pack_voltage': 100., 'cell': {
        'nominal_voltage': 3.6, 'max_cont_discharge_current': 35., 'mass': .0458}}, 20000.)
    assert sized['cells_in_series'] == 28
    assert sized['cells_in_parallel'] == 10
    assert sized['calculated_mass_kg'] == pytest.approx(16.6712)
    assert propulsion.battery_sizing['power_draw_w'] == pytest.approx(
        1000 * sum(p['required_power_kw'] for p in propulsion.pump_sizing.values()))


@pytest.mark.parametrize('key,value', [('pack_voltage', 0), ('pack_voltage', float('nan')),
                                      ('current_fos', .5), ('mass_margin', .9),
                                      ('additional_mass', -1)])
def test_invalid_inputs(design, key, value):
    cfg = deepcopy(design[0]['prop_system']['battery'])
    cfg[key] = value
    with pytest.raises(ValueError):
        size_battery(cfg, 20000)


def test_mass_cg_rebuild_and_overrides(design):
    cfg = deepcopy(design[0])
    original = deepcopy(cfg)
    auto = build(design, cfg)
    i = auto.battery_sizing['section_index']
    mass = auto.battery_sizing['calculated_mass_kg']
    cfg['vehicle']['sections'][i]['masses']['battery'] = 0.
    baseline = build(design, cfg)
    assert auto.dry_mass.sum() - baseline.dry_mass.sum() == pytest.approx(mass)
    battery_station = auto.sections[i].station.mean()
    assert auto.cg == pytest.approx((baseline.cg * baseline.total_mass + mass * battery_station)
                                    / auto.total_mass)
    assert auto.Iyy > baseline.Iyy
    before = auto.dry_mass.copy()
    auto.build(design[3])
    np.testing.assert_allclose(auto.dry_mass, before)
    assert auto.cfg == original
    cfg['vehicle']['sections'][i]['masses']['battery'] = 12.
    manual = build(design, cfg)
    assert manual.dry_mass.sum() - baseline.dry_mass.sum() == pytest.approx(12.)
    assert manual.battery_sizing['mass_source'] == 'battery_override'
    cfg['vehicle']['sections'][i]['mass_override'] = 20.
    overridden = build(design, cfg)
    assert overridden.sections[i].mass.sum() == pytest.approx(20.)
    assert overridden.battery_sizing['used_mass_kg'] is None
    cfg['vehicle']['sections'][i]['masses']['battery'] = 'auto'
    np.testing.assert_allclose(build(design, cfg).dry_mass, overridden.dry_mass)


def test_placement_validation_and_tank_placement(design):
    cfg = deepcopy(design[0])
    source = next(d for d in cfg['vehicle']['sections'] if 'battery' in d.get('masses', {}))
    del source['masses']['battery']
    with pytest.raises(ValueError, match='exactly one'):
        build(design, cfg)
    target = next(d for d in cfg['vehicle']['sections'] if d['type'] == 'prop_tank')
    target.setdefault('masses', {})['battery'] = 'auto'
    vehicle = build(design, cfg)
    tank = vehicle.tanks[target['tank_id']]
    dry = tank.dry_mass.copy()
    vehicle.update_mass_distribution({node: {'axial_mass': np.ones(section.n)}
                                      for node, section, _ in vehicle.fluid_sections})
    np.testing.assert_allclose(tank.mass, dry + 1.)
    source['masses']['battery'] = 'auto'
    with pytest.raises(ValueError, match='exactly one'):
        build(design, cfg)
    del cfg['prop_system']['battery']
    with pytest.raises(ValueError, match='requires prop_system.battery'):
        build(design, cfg)
    source['masses']['battery'] = target['masses']['battery'] = 0.
    assert build(design, cfg).battery_sizing == {}


def test_simulation_and_report(design):
    cfg, pure, combustion, propulsion = design
    cfg = deepcopy(cfg)
    cfg['simulation']['t_end'] = .1
    from diagnostics.errors import EvaluationFailure
    with pytest.raises(EvaluationFailure) as failure:
        simulate(cfg, pure_properties=pure, combustion_properties=combustion,
                 record_history=True)
    assert failure.value.diagnostic['kind'] == 'incomplete_flight'
    result = failure.value.partial_result
    assert result.battery_sizing['used_mass_kg'] == propulsion.battery_sizing['calculated_mass_kg']
    assert any('Electric pump battery' in table for table in build_run_tables(cfg, result, capture_history(result)[1]))


def test_mass_updates_reuse_fixed_layout_and_preserve_engine_overlay(design):
    vehicle = build(design)
    fixed = {name: getattr(vehicle, name) for name in ('station', 'EI', 'lat_area', 'surf_area', 'mass')}
    baseline = vehicle.dry_mass.copy()
    nodes = {node: {'axial_mass': np.ones(section.n)} for node, section, _ in vehicle.fluid_sections}
    vehicle.update_mass_distribution(nodes)
    assert vehicle.total_mass == pytest.approx(baseline.sum() + sum(section.n for _, section, _ in vehicle.fluid_sections))
    for state in nodes.values():
        state['axial_mass'] *= 0.
    vehicle.update_mass_distribution(nodes)
    np.testing.assert_array_equal(vehicle.mass, baseline)
    assert all(getattr(vehicle, name) is value for name, value in fixed.items())
    node, section, _ = vehicle.fluid_sections[0]
    for invalid in (np.full(section.n, -1.), np.full(section.n, np.nan), np.ones(section.n + 1)):
        nodes[node]['axial_mass'] = invalid
        with pytest.raises(ValueError):
            vehicle.update_mass_distribution(nodes)
