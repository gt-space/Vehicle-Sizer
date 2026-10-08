from copy import deepcopy

import numpy as np
import pytest

from Configs.loader import load_config
from Fluids.helpers.battery import size_battery
from Fluids.PropSystem import PropSystem
from Vehicle.Vehicle import Vehicle
from simulation import property_sources, simulate
from reporting.run_report import build_run_tables


@pytest.fixture(scope='module')
def design():
    cfg = load_config('Configs/flight_pump_fed_regulator_epump.yaml')
    # Exercise assembly with a design inside the 15 kW pump limits at 46% efficiency.
    cfg['prop_system']['thrust_target'] = 10000.
    pure, combustion = property_sources(cfg)
    vehicle = Vehicle(cfg, pure)
    propulsion = PropSystem(cfg, vehicle.tanks, fluid_properties=pure,
                            combustion_properties=combustion)
    yield cfg, pure, combustion, propulsion
    propulsion.close()


def build(design, cfg=None):
    original, pure, _, propulsion = design
    vehicle = Vehicle(deepcopy(original if cfg is None else cfg), pure)
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
    tank.set_fluid_mass(np.ones(tank.n))
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
    result = simulate(cfg, pure_properties=pure, combustion_properties=combustion,
                      record_history=True)
    assert result.battery_sizing['used_mass_kg'] == propulsion.battery_sizing['calculated_mass_kg']
    assert result.design_summary['battery_sizing'] == result.battery_sizing
    assert any('Electric pump battery' in table for table in build_run_tables(cfg, result))
