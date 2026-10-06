from copy import deepcopy

import numpy as np
import pytest
import yaml

import optimizer as opt
from Configs.loader import load_config
from Vehicle.Vehicle import Vehicle


@pytest.fixture(scope='module')
def setup():
    model = opt.DragModel()
    cfg, settings = opt.prepare(load_config('Configs/optimizer_pressure_fed.yaml'), model)
    pure, combustion = opt.property_sources(cfg)
    return cfg, settings, model, pure, combustion


@pytest.mark.parametrize('construction', ['copv', 'metal'])
@pytest.mark.parametrize('pressure', [15e6, 30e6])
def test_optimizer_pressurant_geometry_matches_built_vessel(setup, construction, pressure):
    base, settings, _, pure, _ = setup
    cfg = deepcopy(base)
    tank = cfg['tanks']['press_tank']
    tank.update(construction=construction, design_pressure=pressure)
    if construction == 'metal':
        tank.update(material='aluminum_6061', pressure_fos=1.7, weld_allowable=160e6)
    assert 'wall_thickness' not in tank
    vessel = Vehicle(cfg, pure).tanks['press_tank'].vessel
    assert opt.pressurant_inner_radius(cfg, tank) == pytest.approx(vessel.inner_diameter / 2)
    records = opt.primitive_records(cfg, settings['tank_ids'])
    assert records['geometry.copv_cylinder'].margin == pytest.approx(vessel.cylinder_length - 1e-9)
    if construction == 'copv':
        tank['length'] = vessel.length + .2
        vessel = Vehicle(cfg, pure).tanks['press_tank'].vessel
        records = opt.primitive_records(cfg, settings['tank_ids'])
        assert records['geometry.copv_cylinder'].margin == pytest.approx(vessel.cylinder_length - 1e-9)


def test_undersized_copv_is_a_geometry_rejection(setup):
    base, settings, _, _, _ = setup
    cfg = deepcopy(base)
    cfg['tanks']['press_tank']['volume'] = 1e-6
    records = opt.primitive_records(cfg, settings['tank_ids'])
    assert records['geometry.copv_cylinder'].margin < 0


def test_conditioned_candidate_can_be_saved_as_yaml(setup, tmp_path):
    cfg, settings, model, pure, combustion = setup
    evaluator = opt.Evaluator(cfg, settings, model, pure, combustion, tmp_path)
    candidate = evaluator.decode(np.mean(evaluator.bounds, axis=1))
    assert yaml.safe_load(yaml.safe_dump(candidate)) == candidate


@pytest.mark.parametrize('supplied_wall', [None, .02])
def test_conditioned_propellant_minimum_matches_current_wall_sizing(setup, tmp_path, supplied_wall):
    base, original, model, pure, combustion = setup
    cfg, settings = deepcopy(base), deepcopy(original)
    cfg['advanced'].update(tank_pressure_fos=2., weld_allowable=100e6)
    for role in ('oxidizer', 'fuel'):
        settings['bounds'][f'{role}_mass'] = (0., 300.)
        if supplied_wall is not None:
            cfg['tanks'][settings['tank_ids'][role]]['wall_thickness'] = supplied_wall
    evaluator = opt.Evaluator(cfg, settings, model, pure, combustion, tmp_path)
    values = np.mean(evaluator.bounds, axis=1)
    for role in ('oxidizer', 'fuel'):
        values[evaluator.names.index(f'{role}_mass')] = 0.
    candidate = evaluator.decode(values)
    vehicle = Vehicle(candidate, pure)
    # The lower conditioned mass leaves exactly the requested tiny cylinder
    # with the actual tank wall equation, rather than the old yield formula.
    for key in ('ox_tank', 'fuel_tank'):
        assert vehicle.tanks[key].cyl_length == pytest.approx(1e-8, abs=1e-12)
