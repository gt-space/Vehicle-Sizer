import unittest
from copy import deepcopy
import numpy as np
import pytest
from Configs.loader import load_config
from Vehicle.Vehicle import Vehicle
from Vehicle.Material import MaterialProperties
from Vehicle.sections.PressTank import PressTank
from Vehicle.sections.PropTank import PropTank
from FluidTables.PropertyModels import CoolPropPropertySource

class MetalPressurantTests(unittest.TestCase):
    def test_material_and_pressure_sizing(self):
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        tank = cfg['tanks']['press_tank']
        tank.update(construction='metal', material='aluminum_6061', design_pressure=2e6)
        for key in ('equivalent_density', 'thickness_slope', 'thickness_intercept'):
            tank.pop(key)
        vessel = Vehicle(cfg, CoolPropPropertySource()).tanks['press_tank']
        self.assertAlmostEqual(vessel.wall_thickness,
            1.5 * 2e6 * tank['outer_diameter'] / (2 * cfg['advanced']['weld_allowable']))
        self.assertAlmostEqual(vessel._get_dry_mass(), vessel.shell_volume * MaterialProperties.from_name('aluminum_6061').density)
        tank['mass'] = 15
        self.assertEqual(Vehicle(cfg, CoolPropPropertySource()).tanks['press_tank']._get_dry_mass(), 15)


def metal_config():
    cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
    cfg['tanks']['press_tank'].update(construction='metal', material='aluminum_6061',
                                    design_pressure=2e6)
    return cfg


@pytest.mark.parametrize('wall', [None, .01])
def test_metal_pressurant_matches_propellant_tank_sizing(wall):
    cfg = metal_config()
    definition = cfg['tanks']['press_tank']
    cfg['vehicle']['OMLD'] = definition['outer_diameter']
    definition['wall_thickness'] = wall
    press = PressTank(cfg, 'press_tank')
    prop = PropTank(cfg, prop_mass=10., liquid_density=1000., material=press.material,
                    wall_thickness=wall, max_pressure=definition['design_pressure'],
                    t_wall_min=cfg['advanced']['t_wall_min'], passthrough_diameter=0.,
                    passthrough_wall_thickness=0., ellipse_ratio=definition['ellipse_ratio'],
                    ullage_factor=None, volume=definition['volume'], tank_id='comparison')
    assert press.required_wall_thickness == pytest.approx(prop.required_wall_thickness)
    assert press.wall_thickness == pytest.approx(prop.wall_thickness)
    assert press.length == pytest.approx(prop.length)
    assert press._get_dry_mass() == pytest.approx(prop._get_dry_mass())
    before = press._get_dry_mass()
    cfg['advanced']['endcap_mass_multiplier'] *= 2
    assert press._get_dry_mass() > before
    assert press._get_dry_mass() == pytest.approx(prop._get_dry_mass())


def test_metal_length_mass_and_section_overrides_survive_fluid_updates():
    cfg = metal_config()
    cfg['tanks']['press_tank'].update(length=2., mass=15.)
    press = PressTank(cfg, 'press_tank')
    press.start_station = 0.
    press.station = press.local_centers
    press.mass_inputs = {'masses': {'hardware': 3.}}
    press.build()
    assert press.length == 2.
    assert press.dry_mass.sum() == pytest.approx(18.)
    press.set_fluid_mass(np.full(press.n, 2. / press.n))
    assert press.mass.sum() == pytest.approx(20.)
    press.build()
    assert press.mass.sum() == pytest.approx(18.)
    press.mass_inputs['mass_override'] = 12.
    press.build()
    assert press.mass.sum() == pytest.approx(12.)
    assert press.get_thermal_shell_mass().sum() == pytest.approx(12.)
    geometry = press.get_fluid_geometry()
    assert geometry.length == 2.
    assert geometry.volume == press.volume
    assert geometry.axial_mass(2.).sum() == pytest.approx(2.)


def test_supplied_metal_gauge_has_a_geometry_constraint():
    from Vehicle.geometry_constraints import geometry_constraints
    from types import SimpleNamespace
    cfg = metal_config()
    cfg['tanks']['press_tank']['wall_thickness'] = .0001
    with pytest.warns(RuntimeWarning, match='below'):
        press = PressTank(cfg, 'press_tank')
    margins = geometry_constraints(SimpleNamespace(cfg=cfg, sections=[press]))
    assert margins['press_tank.wall_gauge'] < 0
    assert margins['press_tank.airframe'] > 0


def test_volume_units_and_generic_default():
    cfg = metal_config()
    definition = cfg['tanks']['press_tank']
    definition.pop('construction')
    reference = PressTank(cfg, 'press_tank')
    definition['volume_liters'] = definition.pop('volume') * 1000
    press = PressTank(cfg, 'press_tank')
    assert press._get_dry_mass() == pytest.approx(reference._get_dry_mass())
    assert press.length == pytest.approx(reference.length)
    definition['volume'] = .07
    with pytest.raises(ValueError, match='not both'):
        PressTank(cfg, 'press_tank')


@pytest.mark.parametrize('field,value', [('wall_thickness', float('nan')),
                                       ('wall_thickness', 0.), ('length', float('inf')),
                                       ('mass', -1.), ('ellipse_ratio', 1.),
                                       ('pressure_fos', .5), ('construction', 'unknown')])
def test_invalid_pressurant_inputs(field, value):
    cfg = deepcopy(metal_config())
    cfg['tanks']['press_tank'][field] = value
    with pytest.raises(ValueError):
        PressTank(cfg, 'press_tank')
