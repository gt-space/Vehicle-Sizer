import unittest
from copy import deepcopy
import numpy as np
from Configs.loader import load_config
from FluidTables.PropertyModels import CoolPropPropertySource
from Vehicle.Vehicle import Vehicle
from Vehicle.Engine import Engine

class SectionMassTests(unittest.TestCase):
    def test_named_additions_and_overrides_for_every_section(self):
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        vehicle = Vehicle(cfg, CoolPropPropertySource())
        vehicle.engine = Engine(30, .5715, .01)
        vehicle.sections = vehicle._build_sections()
        vehicle._stack_sections()
        for section in vehicle.sections:
            with self.subTest(section=type(section).__name__):
                section.mass_inputs = {}
                section.build()
                baseline = section.mass.sum()
                section.mass_inputs = {'masses': {'arbitrary name': 3, 'another': 4}}
                section.build()
                self.assertAlmostEqual(section.mass.sum(), baseline + 7)
                section.mass_inputs = {'mass_override': 12, 'masses': {'arbitrary': 3}}
                section.build()
                self.assertAlmostEqual(section.mass.sum(), 12)
                np.testing.assert_allclose(section.mass, 12 / section.n)
                if hasattr(section, 'dry_mass'):
                    self.assertAlmostEqual(section.dry_mass.sum(), 12)

    def test_stringers_use_configured_linear_mass(self):
        from Vehicle.sections.InterTank import InterTank
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        sec = InterTank(cfg, .3, 3.6e-5, stringer_unit_mass=2.7)
        sec.get_mass()
        self.assertAlmostEqual(sec.mass.sum(), 2.7 * .3 * cfg['inter_tank']['stringer_count'])

    def test_pressure_fos_allowable_and_endcap_multiplier(self):
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        vehicle = Vehicle(cfg, CoolPropPropertySource())
        tank = vehicle.tanks['ox_tank']
        self.assertAlmostEqual(tank.wall_thickness,
            cfg['advanced']['tank_pressure_fos'] * tank.max_pressure * tank.OMLD / 2
            / cfg['advanced']['weld_allowable'])
        base = tank._get_dry_mass()
        a = tank.OMLD / 2
        c = a / tank.ellipse_ratio
        area = 4 * np.pi * ((a**3.2 + 2 * (a*c)**1.6) / 3)**(1/1.6)
        cfg['advanced']['endcap_mass_multiplier'] += 1
        self.assertAlmostEqual(tank._get_dry_mass() - base,
                               area * tank.wall_thickness * tank.material.density)

    def test_copv_correlation_and_override(self):
        from Vehicle.sections.PressTank import PressTank
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        copv = PressTank(cfg, 'press_tank')
        self.assertAlmostEqual(copv.wall_thickness, .03 * .2794 + .004)
        self.assertAlmostEqual(copv._get_dry_mass(), copv.shell_volume * 2238.6)
        cfg['tanks']['press_tank']['mass'] = 18
        self.assertEqual(PressTank(cfg, 'press_tank')._get_dry_mass(), 18)
