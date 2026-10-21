import unittest
from types import SimpleNamespace
from unittest.mock import patch
from Vehicle.sections.FinCan import FinCan
import numpy as np
from Configs.loader import load_config
from Vehicle.Vehicle import Vehicle
from Vehicle.sections.InterTank import InterTank
from Vehicle.sections.Nosecone import Nosecone

class SectionStiffnessTests(unittest.TestCase):
    def test_intertank_uses_input_independent_of_mass(self):
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        for mass in [1, 100]:
            section = InterTank(cfg, .3, 123456)
            section.mass_inputs = {"masses": {"stringers": mass}}
            section.get_mass()
            section.apply_mass_inputs()
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
        vehicle.cfg = {'engine': {'mass': 1., 'length': .1}, 'vehicle': {'OMLD': .25}}
        vehicle.tanks = {}
        sections = [SimpleNamespace(n=1, station=np.array([i + .5]), mass=np.array([1.]),
                    EI=np.array([value]), lat_area=np.ones(1), surf_area=np.ones(1),
                    cg=i + .5, Ixx=0., Iyy=0.)
                    for i, value in enumerate([11, 22, 33])]
        fin = FinCan.__new__(FinCan)
        fin.__dict__.update(sections[0].__dict__)
        sections[0] = fin
        def stack():
            for i, section in enumerate(vehicle.sections):
                section.start_station = float(i)
                section.cell_edges = np.array([i, i + 1.])
            vehicle.length = 3.
        for order in [sections, sections[::-1], [sections[1], sections[0], sections[2]]]:
            with patch.object(vehicle, '_build_sections', return_value=order), \
                 patch.object(vehicle, '_stack_sections', side_effect=stack):
                vehicle.build(SimpleNamespace(exit_area=.01, node_definitions={}))
            np.testing.assert_array_equal(vehicle.EI, [s.EI[0] for s in order])
