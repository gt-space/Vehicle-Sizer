"""Continuous conservation and component ownership checks."""

import numpy as np
import pytest

from Fluids import BranchState, FluidNetwork, FluidState, NodeState
from Fluids.FluidNode import inventory_residual
from Fluids.PropSystem import PropSystem


def test_continuous_balance_accounts_for_flow_heat_and_expansion_work():
    def branch(mdot, enthalpy):
        return BranchState(flows={"main": {
            "mdot": mdot, "direction": 1,
            "fluid": FluidState("Nitrogen", "gas", {"h": enthalpy}),
        }})

    # Inflow 2 kg/s at 10 J/kg; outflow 1 kg/s at 4 J/kg.
    # Add 3 W of heat, do 5 W of work: dm/dt=1 kg/s, dU/dt=14 W.
    args = dict(adjacent=[(1, branch(2, 10)), (-1, branch(1, 4))],
                fluid="Nitrogen", heat_rate=3, pressure=5, volume_rate=1)
    np.testing.assert_allclose(inventory_residual(mass_rate=1, energy_rate=14, **args), 0)
    np.testing.assert_allclose(inventory_residual(mass_rate=1, energy_rate=15, **args), [0, 1])


def test_phase_interface_work_cancels_in_total_energy():
    liquid = inventory_residual(mass_rate=0, energy_rate=6, adjacent=[],
                                pressure=3, volume_rate=-2)
    gas = inventory_residual(mass_rate=0, energy_rate=-6, adjacent=[],
                             pressure=3, volume_rate=2)
    np.testing.assert_allclose(liquid, 0)
    np.testing.assert_allclose(gas, 0)


def test_network_copies_definitions():
    nodes = {"tank": {"component": "boundary", "P": 1e5}}
    network = FluidNetwork(nodes, {}, fluid_properties=object())
    network.node_definitions["tank"]["P"] = 2e5
    assert nodes["tank"]["P"] == 1e5
    assert network.nodes["tank"].definition["P"] == 1e5
