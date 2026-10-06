import unittest
from types import SimpleNamespace
import numpy as np
from Configs.loader import load_config
from Vehicle.Vehicle import Vehicle
from Vehicle.sections.InterTank import InterTank
from Vehicle.sections.Nosecone import Nosecone

class SectionStiffnessTests(unittest.TestCase):
    def test_intertank_uses_input_independent_of_mass(self):
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        for unit_mass in [1, 100]:
            section = InterTank(cfg, .3, 123456, stringer_unit_mass=unit_mass)
            section.get_mass()
            section.build_stiffness()
            np.testing.assert_array_equal(section.EI, np.full(section.n, 123456))

    def test_generic_override_and_validation(self):
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        section = Nosecone(cfg)
        section.stiffness_input = 654321
        section.build_stiffness()
        np.testing.assert_array_equal(section.EI, np.full(section.n, 654321))
        for value in [0, -1, float('nan'), float('inf')]:
            section.stiffness_input = value
            with self.assertRaises(ValueError):
                section.build_stiffness()

    def test_assembly_follows_object_order(self):
        vehicle = Vehicle.__new__(Vehicle)
        sections = [SimpleNamespace(station=np.array([i]), mass=np.array([1]),
                    EI=np.array([value]), lat_area=np.array([1]), surf_area=np.array([1]))
                    for i, value in enumerate([11, 22, 33])]
        for order in [sections, sections[::-1], [sections[1], sections[0], sections[2]]]:
            vehicle.sections = order
            vehicle._assemble_vectors()
            np.testing.assert_array_equal(vehicle.EI, [s.EI[0] for s in order])
