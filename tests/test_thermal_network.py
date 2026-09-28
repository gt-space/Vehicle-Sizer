from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from simulation_types import AeroOut, AtmosState, KinematicsState
from Thermals import ThermalNetwork
from Thermals.ThermalCircuit import solve_lumped_wall
from Thermals.ThermalNode import natural_convection_htc


class WetSection:
    tank_id = "tank"
    n = 4
    start_station = 0.0
    end_station = 4.0
    station = np.arange(4.0)
    dx = 1.0
    wall_thickness = 0.01
    wall_material = "aluminum_6061"
    emissivity = 0.0
    cfg = {}

    @staticmethod
    def get_thermal_oml_area():
        return np.ones(4)

    @staticmethod
    def get_thermal_internal_area():
        return np.ones(4)

    @staticmethod
    def get_thermal_shell_mass():
        return np.ones(4)


def test_lumped_wall_stores_applied_energy():
    wall_T = solve_lumped_wall(
        dt=2.0,
        wall_T=np.array([300.0]),
        capacitance=np.array([10.0]),
        heat_source=np.array([50.0]),
        conductance=np.array([0.0]),
        boundary_T=np.array([0.0]),
    )

    np.testing.assert_allclose(wall_T, [310.0])


def test_natural_convection_htc_increases_with_grashof_number():
    common = {
        "length": np.array([1.0]),
        "rho": np.array([2.0]),
        "mu": np.array([1.8e-5]),
        "conductivity": np.array([0.026]),
        "specific_heat": np.array([1000.0]),
        "expansion": np.array([1.0 / 300.0]),
    }
    low_h, low_gr = natural_convection_htc(
        np.array([5.0]), acceleration=1.0, **common
    )
    high_h, high_gr = natural_convection_htc(
        np.array([10.0]), acceleration=10.0, **common
    )

    assert high_gr[0] > low_gr[0]
    assert high_h[0] > low_h[0]


def fluid_output():
    return SimpleNamespace(
        node={
            "tank": {
                "tank_id": "tank",
                "fill_height": 1.6,
                "fluids": {
                    "propellant": {
                        "phase": "liquid", "T": 290.0, "rho": 800.0,
                        "mu": 1.0e-3, "k": 0.12, "cp": 2000.0, "beta": 8.0e-4,
                    },
                    "ullage": {
                        "phase": "gas", "T": 280.0, "rho": 2.0,
                        "mu": 1.8e-5, "k": 0.026, "cp": 1000.0,
                        "beta": 1.0 / 280.0,
                    },
                },
            }
        }
    )


def inputs():
    kin = KinematicsState(t=0.0, dt=1.0, x=0., h=0.0, vx=0., vz=0.0, theta=np.pi/2, q=0.0, alpha=0.0, m=1.0, Iyy=1.0)
    atm = AtmosState(300.0, 101325.0, 1.2, 1.8e-5, 340.0, 0.0, 0.0)
    return kin, atm, AeroOut(0.0, 0.0)


@patch("Thermals.ThermalNode.heating.get_recovery_temperature", return_value=300.0)
@patch("Thermals.ThermalNode.heating.get_body_heating", return_value=np.zeros(4))
def test_wet_node_rounds_fill_to_cells_and_commits_once(*_):
    cfg = {
        "tanks": {"tank": {"thermal": {"model": "Aeroheating"}}},
        "thermal": {
            "initial_temperature": 300.0,
            "sink_temperature": 300.0,
            "material_overrides": {
                "aluminum_6061": {
                    "specific_heat": 900.0,
                    "thermal_conductivity": 150.0,
                }
            },
        }
    }
    network = ThermalNetwork(cfg, SimpleNamespace(sections=[WetSection()]))
    thermal_out = network.trial(
        *inputs(), fluid_output(), axial_specific_force=9.81
    )
    output = thermal_out.node["tank"]

    np.testing.assert_array_equal(
        output["cells"]["phase"], ["gas", "gas", "liquid", "liquid"]
    )
    assert output["phases"]["liquid"]["heat_rate"] > 0.0
    assert output["phases"]["gas"]["heat_rate"] > 0.0
    assert output["phases"]["liquid"]["htc"] > 0.0
    assert output["phases"]["gas"]["grashof"] > 0.0
    assert output["cells"]["internal_htc"].shape == (4,)
    assert output["cells"]["grashof"].shape == (4,)
    assert output["cells"]["biot"].shape == (4,)
    np.testing.assert_array_equal(network.wall_T, np.full(4, 300.0))

    network.commit(thermal_out)
    assert np.all(network.wall_T < 300.0)


@patch("Thermals.ThermalNode.heating.get_recovery_temperature", return_value=300.0)
@patch("Thermals.ThermalNode.heating.get_body_heating", return_value=np.zeros(4))
def test_omitted_model_builds_no_wall_nodes(*_):
    cfg = {
        "thermal": {
            "initial_temperature": 300.0,
            "sink_temperature": 300.0,
            "material_overrides": {
                "aluminum_6061": {
                    "specific_heat": 900.0,
                    "thermal_conductivity": 150.0,
                }
            },
        }
    }
    output = ThermalNetwork(
        cfg, SimpleNamespace(sections=[WetSection()])
    ).trial(*inputs(), fluid_output())

    assert output.heat_rates() == {}
    assert output.node == {}
