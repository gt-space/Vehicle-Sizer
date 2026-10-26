"""Manufacturing allowance affects geometry, not the optimizer's calculated-wall minimum."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from Configs.loader import load_config
from FluidTables.PropertyModels import property_sources
from Fluids.design import prepare_design_inputs
from Vehicle.Vehicle import Vehicle
from Vehicle.sections.PressTank import PressTank
from diagnostics.constraints import vehicle_limit_records


@pytest.fixture(scope='module')
def setup():
    cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
    pure, _ = property_sources(cfg)
    return cfg, pure


def tanks(cfg, pure):
    _, states, walls = prepare_design_inputs(cfg)
    return Vehicle(cfg, pure, initial_states=states, wall_pressures=walls).tanks


def test_allowance_changes_geometry_but_not_required_wall_and_overrides_are_final(setup):
    cfg, pure = setup
    cfg = deepcopy(cfg)
    cfg['advanced']['tank_wall_thickness_allowance'] = 0.
    base = tanks(cfg, pure)
    cfg['advanced']['tank_wall_thickness_allowance'] = .000254
    padded = tanks(cfg, pure)
    for name in ('ox_tank', 'fuel_tank'):
        assert padded[name].required_wall_thickness == base[name].required_wall_thickness
        assert padded[name].wall_thickness == pytest.approx(base[name].wall_thickness + .000254)
        assert padded[name].length > base[name].length
        assert padded[name]._get_dry_mass() > base[name]._get_dry_mass()
        cfg['tanks'][name]['wall_thickness'] = .003
    assert padded['press_tank'].wall_thickness == base['press_tank'].wall_thickness
    for name in ('ox_tank', 'fuel_tank'):
        assert tanks(cfg, pure)[name].wall_thickness == .003


@pytest.mark.parametrize('required,expected', [(.0024, -.00014), (.00254, 0.), (.0026, .00006)])
def test_minimum_uses_calculated_wall_before_allowance(required, expected):
    cfg = {'tanks': {'ox': {'min_wall_thickness': .00254}}}
    vehicle = SimpleNamespace(tanks={'ox': SimpleNamespace(required_wall_thickness=required,
                                                         wall_thickness=required + .000254)})
    record = vehicle_limit_records(cfg, vehicle)['tank.ox.wall_min']
    assert record.margin == pytest.approx(expected)
    assert record.units == 'm'
    assert record.required
    cfg['tanks']['ox'].clear()
    assert vehicle_limit_records(cfg, vehicle) == {}


def test_metal_pressurant_allowance_and_explicit_override(setup):
    cfg, _ = setup
    definition = dict(construction='metal', outer_diameter=.3, design_pressure=3e6)
    wall, required = PressTank.wall_sizing(cfg, definition)
    assert wall - required == pytest.approx(.000254)
    definition['wall_thickness'] = .01
    assert PressTank.wall_sizing(cfg, definition) == (.01, required)


@pytest.mark.parametrize('allowance', [-.001, float('nan'), float('inf')])
def test_invalid_allowance_is_rejected(setup, allowance):
    cfg, pure = setup
    cfg = deepcopy(cfg)
    cfg['advanced']['tank_wall_thickness_allowance'] = allowance
    with pytest.raises(ValueError, match='allowance'):
        tanks(cfg, pure)


def test_optimizer_config_applies_minimum_to_both_propellant_tanks(setup):
    from optimizer import core as opt
    cfg, settings = opt.prepare(load_config('Configs/optimizer_pressure_fed.yaml'), opt.DragModel())
    _, pure = setup
    built = tanks(cfg, pure)
    records = vehicle_limit_records(cfg, SimpleNamespace(tanks=built, length=9., cfg=cfg))
    for name in ('ox_tank', 'fuel_tank'):
        assert cfg['tanks'][name]['min_wall_thickness'] == .00254
        assert records[f'tank.{name}.wall_min'].margin < 0
        assert built[name].wall_thickness == pytest.approx(built[name].required_wall_thickness + .000254)
