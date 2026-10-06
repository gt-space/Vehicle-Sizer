import unittest
from Configs.loader import load_config
from Vehicle.Vehicle import Vehicle
from Vehicle.Material import MaterialProperties
from FluidTables.PropertyModels import CoolPropPropertySource

class MetalPressurantTests(unittest.TestCase):
    def test_material_and_pressure_sizing(self):
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        tank = cfg['tanks']['press_tank']
        tank.update(construction='metal', material='aluminum_6061', design_pressure=2e6)
        for key in ('equivalent_density', 'thickness_slope', 'thickness_intercept'):
            tank.pop(key)
        vessel = Vehicle(cfg, CoolPropPropertySource()).tanks['press_tank'].vessel
        self.assertAlmostEqual(vessel.wall_thickness,
            1.5 * 2e6 * tank['outer_diameter'] / (2 * cfg['advanced']['weld_allowable']))
        self.assertAlmostEqual(vessel.mass, vessel.shell_volume * MaterialProperties.from_name('aluminum_6061').density)
        tank['mass'] = 15
        self.assertEqual(Vehicle(cfg, CoolPropPropertySource()).tanks['press_tank'].vessel.mass, 15)
