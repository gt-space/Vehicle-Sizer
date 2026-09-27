import numpy as np
import pytest

from Flight.Flight import FlightSim
from simulation_types import AeroOut, AtmosState, KinematicsState, ThermalOut
from tests.test_flight import FakeAero, FakeEnvironment, FakePropSystem, FakeVehicle


class FakeThermal:
    def __init__(self):
        self.calls = []

    def trial(
        self, kin, atm, aero, fluids, previous=None, axial_specific_force=0.0
    ):
        del axial_specific_force
        self.calls.append(fluids)
        return ThermalOut(
            node={
                "copv": {
                    "cells": {"wall_T": np.array([300.0])},
                    "phases": {"gas": {"heat_rate": 10.0}},
                },
                "tank": {
                    "cells": {"wall_T": np.array([300.0])},
                    "phases": {
                        "liquid": {"heat_rate": 20.0},
                        "gas": {"heat_rate": 5.0},
                    },
                },
            },
        )

    def commit(self, thermal_out):
        self.committed = thermal_out



def test_flight_converges_node_heat_boundaries_before_fluid_commit():
    propulsion = FakePropSystem()
    thermal = FakeThermal()
    flight = FlightSim(
        cfg={
            "simulation": {"dt": 0.1, "t_end": 0.1},
            "launch": {"altitude": 0.0, "rail_height": 5.0},
            "advanced": {
                "heating": {"convergence_tolerance": 1.0e-3, "max_iterations": 2}
            },
        },
        env=FakeEnvironment(),
        aero=FakeAero(),
        prop_system=propulsion,
        vehicle=FakeVehicle(),
        thermal=thermal,
    )
    atmosphere = AtmosState(288.0, 101325.0, 1.2, 1.8e-5, 340.0, 0.0, 2.0)
    initial = propulsion.update(None, atmosphere, {}, commit=False)
    kin = KinematicsState(0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 10.0, 2.0)
    aero = AeroOut(0.0, 0.0)

    _, thermal_out = flight.step_coupled(kin, atmosphere, aero, initial, 0.0)

    assert propulsion.calls[-2]["commit"] is False
    assert propulsion.calls[-1]["commit"] is True
    assert propulsion.calls[-1]["heat_rate"] == thermal_out.heat_rates()
    assert thermal.committed is thermal_out


def test_failed_fluid_preview_is_not_silently_replaced_by_a_commit():
    class FailingPreviewPropSystem(FakePropSystem):
        def update(self, dt, atm, heat_rate=None, commit=True, **kwargs):
            if dt is not None and not commit:
                raise RuntimeError("preview failure")
            return super().update(dt, atm, heat_rate, commit, **kwargs)

    propulsion = FailingPreviewPropSystem()
    thermal = FakeThermal()
    flight = FlightSim(
        cfg={"advanced": {"heating": {"max_iterations": 2}}},
        env=FakeEnvironment(),
        aero=FakeAero(),
        prop_system=propulsion,
        vehicle=FakeVehicle(),
        thermal=thermal,
    )
    atmosphere = AtmosState(288.0, 101325.0, 1.2, 1.8e-5, 340.0, 0.0, 0.0)
    initial = propulsion.update(None, atmosphere, {}, commit=False)
    kin = KinematicsState(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 10.0, 2.0)

    with pytest.raises(RuntimeError, match="preview failure"):
        flight.step_coupled(
            kin, atmosphere, AeroOut(0.0, 0.0), initial, 9.81
        )
    assert not hasattr(thermal, "committed")
