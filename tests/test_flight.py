import math
import unittest

import numpy as np

from Flight.Flight import FlightSim
from Flight.flight_forces import gravity
from Flight.types import (
    AeroOut,
    AtmosState,
    FluidOut,
    KinematicsState,
    PlantOut,
    PropulsionOut,
    ThermalOut,
)


class FakeEnvironment:
    @staticmethod
    def atmosphere(altitude, airspeed):
        return AtmosState(
            T=288.0,
            p=101325.0,
            rho=1.2,
            mu=1.8e-5,
            a=340.0,
            q=altitude,
            Ma=abs(airspeed) / 340.0,
        )


class FakeAero:
    reference_area = 1.0

    @staticmethod
    def evaluate(kinematics, atmosphere, engine_on):
        del atmosphere, engine_on
        normal = 20.0 * kinematics.alpha
        return AeroOut(
            Cd=0.0,
            D=0.0,
            Ca=0.0,
            A=0.0,
            Cn=normal,
            N=normal,
            cp=1.5,
        )


class FakePropSystem:
    def __init__(self):
        self.calls = []

    def update(
        self,
        dt,
        atm,
        heat_rate,
        bcs=None,
        commit=True,
        axial_specific_force=0.0,
    ):
        self.calls.append(
            {
                "dt": dt,
                "atm": atm,
                "heat_rate": heat_rate,
                "bcs": bcs,
                "commit": commit,
                "axial_specific_force": axial_specific_force,
            }
        )
        liquid_mass = 5.0 if dt is None else 4.9
        return FluidOut(
            node={
                "tank": {
                    "mass": liquid_mass,
                    "tank_id": "tank",
                    "axial_mass": [liquid_mass],
                },
                "copv": {
                    "mass": 1.0,
                    "tank_id": "copv",
                    "axial_mass": [1.0],
                },
                "junction": {"P": 2.0e6},
            },
            branch={},
            td_state={"tank": {}, "copv": {}},
            mdot={},
            propulsion=PropulsionOut(
                mode="combusting",
                shutdown_reason=None,
                thrust=300.0,
                Pc=2.0e6,
                MR=2.0,
                Cf=1.5,
                cstar=1500.0,
                mdot_ox=1.0,
                mdot_fuel=0.5,
                mdot_nozzle=1.5,
            ),
        )


class FakeVehicle:
    def __init__(self):
        self.total_mass = 10.0
        self.Ixx = 2.0
        self.Iyy = 3.0
        self.cg = 1.0
        self.length = 2.0
        self.station = np.array([0.0, 1.0])
        self.mass = np.array([5.0, 5.0])

    def update_mass_distribution(self, node_states):
        self.total_mass = 10.0 + sum(
            sum(state["axial_mass"])
            for state in node_states.values()
            if "axial_mass" in state
        )
        self.mass = np.array([5.0, self.total_mass - 5.0])
        self.cg = float(np.sum(self.mass * self.station) / self.total_mass)


class FlightSkeletonTests(unittest.TestCase):
    def setUp(self):
        self.prop_system = FakePropSystem()
        self.flight = FlightSim(
            cfg={
                "simulation": {"dt": 0.1, "t_end": 0.1},
                "launch": {"altitude": 0.0, "rail_height": 5.0},
            },
            env=FakeEnvironment(),
            aero=FakeAero(),
            prop_system=self.prop_system,
            vehicle=FakeVehicle(),
        )

    def run_flight(self):
        return self.flight.run(compute_loads=False)

    def test_run_uses_prop_system_and_dynamic_node_mass(self):
        history = self.run_flight()

        self.assertEqual(len(history), 1)
        self.assertEqual(self.prop_system.calls[0]["dt"], None)
        self.assertTrue(self.prop_system.calls[0]["commit"])
        self.assertEqual(self.prop_system.calls[1]["dt"], 0.1)
        self.assertTrue(self.prop_system.calls[1]["commit"])
        self.assertAlmostEqual(
            self.prop_system.calls[0]["axial_specific_force"],
            gravity(10.0, 0.0) / 10.0,
        )
        self.assertAlmostEqual(
            self.prop_system.calls[1]["axial_specific_force"], 300.0 / 16.0
        )
        self.assertAlmostEqual(history[0]["kinematics"].m, 15.9)
        start_acceleration = (300.0 - gravity(16.0, 0.0)) / 16.0
        end_acceleration = history[0]["forces"]["az"]
        self.assertAlmostEqual(
            history[0]["kinematics"].vz,
            0.5 * (start_acceleration + end_acceleration) * 0.1,
        )
        self.assertAlmostEqual(
            history[0]["kinematics"].h,
            0.5 * history[0]["kinematics"].vz * 0.1,
        )

    def test_history_is_synchronized_at_end_of_step(self):
        history = self.run_flight()
        result = history[0]
        kin = result["kinematics"]

        self.assertAlmostEqual(result["atmosphere"].q, kin.h)
        self.assertEqual(len(self.prop_system.calls), 2)
        self.assertAlmostEqual(result["plant"].fluids.node["tank"]["mass"], 4.9)
        self.assertTrue(result["on_rail"])
        self.assertEqual(kin.alpha, 0.0)

    def test_rail_constraint_locks_lateral_and_pitch_states(self):
        kin = KinematicsState(
            t=1.0,
            dt=0.1,
            x=2.0,
            h=1.0,
            vx=3.0,
            vz=4.0,
            theta=0.4,
            q=0.5,
            alpha=0.2,
            m=10.0,
            Iyy=3.0,
        )

        constrained = self.flight.constrain_to_rail(kin)

        self.assertEqual(constrained.x, 0.0)
        self.assertEqual(constrained.vx, 0.0)
        self.assertAlmostEqual(constrained.theta, np.pi / 2)
        self.assertEqual(constrained.q, 0.0)
        self.assertEqual(constrained.alpha, 0.0)
        self.assertEqual(constrained.h, kin.h)
        self.assertEqual(constrained.vz, kin.vz)

    def test_flight_kinematics_derives_and_wraps_angle_of_attack(self):
        speed = 100.0
        gamma = np.deg2rad(179.0)
        kin = KinematicsState(
            t=0.0,
            dt=0.1,
            x=0.0,
            h=0.0,
            vx=speed * np.cos(gamma),
            vz=speed * np.sin(gamma),
            theta=np.deg2rad(-179.0),
            q=0.0,
            alpha=0.0,
            m=10.0,
            Iyy=3.0,
        )

        inertial_speed, inertial_gamma, airspeed, gamma_air, alpha = self.flight.flight_kinematics(kin)

        self.assertAlmostEqual(inertial_speed, speed)
        self.assertAlmostEqual(airspeed, speed)
        self.assertAlmostEqual(inertial_gamma, gamma)
        self.assertAlmostEqual(gamma_air, gamma)
        self.assertAlmostEqual(alpha, np.deg2rad(2.0))

    def test_corrector_updates_all_planar_states(self):
        kin = KinematicsState(
            t=0.0,
            dt=0.2,
            x=1.0,
            h=2.0,
            vx=3.0,
            vz=4.0,
            theta=0.5,
            q=0.6,
            alpha=0.0,
            m=10.0,
            Iyy=3.0,
        )
        start = {"ax": 1.0, "az": 2.0, "pitch_acceleration": 3.0}
        end = {"ax": 5.0, "az": 6.0, "pitch_acceleration": 7.0}

        corrected = self.flight.correct_kinematics(kin, start, end, 9.0, 4.0)

        expected_vx = 3.0 + 0.5 * (1.0 + 5.0) * 0.2
        expected_vz = 4.0 + 0.5 * (2.0 + 6.0) * 0.2
        expected_q = 0.6 + 0.5 * (3.0 + 7.0) * 0.2
        self.assertAlmostEqual(corrected.vx, expected_vx)
        self.assertAlmostEqual(corrected.vz, expected_vz)
        self.assertAlmostEqual(corrected.x, 1.0 + 0.5 * (3.0 + expected_vx) * 0.2)
        self.assertAlmostEqual(corrected.h, 2.0 + 0.5 * (4.0 + expected_vz) * 0.2)
        self.assertAlmostEqual(corrected.q, expected_q)
        self.assertAlmostEqual(corrected.theta, 0.5 + 0.5 * (0.6 + expected_q) * 0.2)
        self.assertEqual(corrected.m, 9.0)
        self.assertEqual(corrected.Iyy, 4.0)

    def test_forces_transform_body_forces_and_restore_pitch(self):
        kin = KinematicsState(
            t=0.0,
            dt=0.1,
            x=0.0,
            h=0.0,
            vx=1.0,
            vz=0.0,
            theta=0.0,
            q=0.0,
            alpha=0.1,
            m=10.0,
            Iyy=5.0,
        )
        plant = PlantOut(
            aero=AeroOut(Cd=0.1, D=12.0, Ca=0.1, A=10.0, Cn=0.2, N=20.0, cp=2.0),
            thermal=None,
            fluids=FluidOut(
                node={}, branch={}, td_state={}, mdot={},
                propulsion=PropulsionOut(
                    mode="combusting", shutdown_reason=None, thrust=100.0,
                    Pc=1.0, MR=1.0, Cf=1.0, cstar=1.0,
                    mdot_ox=0.0, mdot_fuel=0.0, mdot_nozzle=0.0,
                ),
            ),
        )
        self.flight.vehicle.cg = 1.0

        forces = self.flight.forces(kin, plant, kin.m)

        self.assertAlmostEqual(forces["Fx"], 90.0)
        self.assertAlmostEqual(forces["Fz"], 20.0 - gravity(10.0, 0.0))
        self.assertAlmostEqual(forces["axial_specific_force"], 9.0)
        self.assertAlmostEqual(forces["drag"], 12.0)
        self.assertAlmostEqual(forces["pitch_moment"], -20.0)
        self.assertAlmostEqual(forces["pitch_acceleration"], -4.0)

    def test_engine_on_uses_combustion_mode_not_residual_thrust(self):
        propulsion = self.prop_system.update(
            dt=None,
            atm=FakeEnvironment.atmosphere(0.0, 0.0),
            heat_rate={},
            commit=False,
        ).propulsion
        self.assertTrue(self.flight.engine_on(propulsion))

        propulsion.mode = "shutdown"
        propulsion.thrust = 50.0

        self.assertFalse(self.flight.engine_on(propulsion))

    def test_history_records_end_forces_mass_and_planar_result(self):
        result = self.run_flight()[0]
        kin = result["kinematics"]
        mass = result["mass_properties"]
        forces = result["forces"]

        self.assertAlmostEqual(mass["total_mass"], kin.m)
        self.assertAlmostEqual(mass["cg"], self.flight.vehicle.cg)
        np.testing.assert_array_equal(mass["station"], self.flight.vehicle.station)
        np.testing.assert_array_equal(mass["axial_mass"], self.flight.vehicle.mass)
        self.assertAlmostEqual(forces["gravity"], gravity(kin.m, kin.h))
        self.assertAlmostEqual(forces["az"], forces["Fz"] / kin.m)
        self.assertAlmostEqual(self.flight.result.final_x, kin.x)
        self.assertAlmostEqual(self.flight.result.final_vx, kin.vx)
        self.assertAlmostEqual(self.flight.result.final_vz, kin.vz)
        self.assertAlmostEqual(self.flight.result.final_speed, math.hypot(kin.vx, kin.vz))
        self.assertAlmostEqual(self.flight.result.final_pitch_angle, kin.theta)
        self.assertAlmostEqual(self.flight.result.final_pitch_rate, kin.q)
        self.assertAlmostEqual(self.flight.result.final_velocity, kin.vz)

    def test_trial_fluid_passes_node_heat_rate(self):
        atmosphere = FakeEnvironment.atmosphere(0.0, 0.0)
        thermal = ThermalOut(
            node={
                "wall": {
                    "fluid_node_id": "tank",
                    "phases": {"liquid": {"heat_rate": 25.0}},
                }
            }
        )

        self.flight.trial_fluid(
            0.1,
            atmosphere,
            thermal_out=thermal,
            axial_specific_force=12.0,
        )

        self.assertEqual(self.prop_system.calls[-1]["heat_rate"], {"tank": {"liquid": 25.0}})
        self.assertFalse(self.prop_system.calls[-1]["commit"])
        self.assertEqual(self.prop_system.calls[-1]["axial_specific_force"], 12.0)


if __name__ == "__main__":
    unittest.main()
