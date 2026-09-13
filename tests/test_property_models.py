from dataclasses import dataclass
from pathlib import Path

import pytest

from Flight.FluidNetwork import FluidNetwork
from Flight.FluidNode import CombustorComponent
from Flight.FluidState import BranchState, FluidState
from FluidProperties.PropertyModels import (
    CombustionProperties,
    PureFluidProperties,
    TableCombustionPropertySource,
    TablePureFluidPropertySource,
)


def branch_flow(branch_id, incidence, state):
    del branch_id
    return incidence, BranchState(
        enabled=state["enabled"],
        flows={
            name: {
                "mdot": values["mdot"],
                "direction": 1,
                "fluid": FluidState.from_dict(name, values),
            }
            for name, values in state["components"].items()
        },
    )


@dataclass(frozen=True)
class GasGeometry:
    volume: float
    internal_area: float = 1.0

    @staticmethod
    def axial_mass(mass):
        return [mass]


class FixedFluidProperties:
    def __init__(self):
        self.calls = 0

    def state_pt(self, fluid, pressure, temperature):
        self.calls += 1
        return PureFluidProperties(
            P=pressure,
            T=temperature,
            rho=2.0,
            h=400_000.0,
            u=300_000.0,
            R=296.8,
            gamma=1.4,
        )


class FixedCombustionProperties:
    def __init__(self):
        self.calls = 0

    def evaluate(self, **conditions):
        self.calls += 1
        return CombustionProperties(
            cstar=1500.0,
            Cf=1.5,
            R=350.0,
            gamma=1.2,
            T=3000.0,
        )


def test_gas_node_accepts_an_injected_property_source():
    properties = FixedFluidProperties()
    network = FluidNetwork(
        nodes={
            "tank": {
                "component": "pressurant_tank",
                "fluid": "test_gas",
                "geometry": GasGeometry(volume=1.0),
                "state0": {
                    "P": 200_000.0,
                    "T": 300.0,
                    "m": 2.0,
                    "U": 600_000.0,
                },
            }
        },
        branches={},
        fluid_properties=properties,
    )

    result = network.update()

    assert properties.calls == 1
    assert result["node"]["tank"]["P"] == 200_000.0
    assert result["node"]["tank"]["fluids"]["test_gas"]["rho"] == 2.0


def test_combustion_node_accepts_an_injected_property_source():
    properties = FixedCombustionProperties()
    node = CombustorComponent(
        "chamber",
        {
            "P0": 2.0e6,
            "oxidizer_fluid": "ox",
            "fuel_fluid": "fuel",
            "combustion_fluid": "products",
            "ambient_node": "ambient",
            "expansion_ratio": 5.0,
            "cstar_efficiency": 1.0,
            "cf_efficiency": 1.0,
        },
    )
    node.combustion_properties = properties
    adjacent = [
        branch_flow(
            "ox",
            1.0,
            {
                "components": {"ox": {"phase": "liquid", "mdot": 2.0}},
                "enabled": True,
            },
        ),
        branch_flow(
            "fuel",
            1.0,
            {
                "components": {"fuel": {"phase": "liquid", "mdot": 1.0}},
                "enabled": True,
            },
        ),
    ]

    result = node.trial_state(
        {"P": 2.0e6}, adjacent, {"ambient": {"P": 100_000.0}}
    )

    assert properties.calls == 1
    assert result["MR"] == pytest.approx(2.0)
    assert result["cstar"] == 1500.0
    assert result["fluids"]["products"]["T"] == 3000.0


def test_table_combustion_source_sizes_and_evaluates_engine():
    root = Path(__file__).resolve().parents[1]
    source = TableCombustionPropertySource(
        root / "FluidProperties" / "sizer_lookups.h5", nfz=1
    )

    expansion_ratio = source.expansion_ratio(2.0e6, 2.0, 101_325.0)
    properties = source.evaluate(
        chamber_pressure=2.0e6,
        mixture_ratio=2.0,
        ambient_pressure=101_325.0,
        expansion_ratio=expansion_ratio,
        cstar_efficiency=0.95,
        cf_efficiency=0.95,
    )

    assert 1.0 < expansion_ratio < 10.0
    assert properties.cstar > 0.0
    assert properties.Cf > 0.0
    assert properties.R > 0.0
    assert properties.gamma > 1.0
    assert properties.T > 0.0


def test_table_fluid_source_evaluates_pt_without_inversion():
    root = Path(__file__).resolve().parents[1]
    source = TablePureFluidPropertySource(
        root / "FluidProperties" / "sizer_lookups.h5",
        {
            "Nitrogen": {
                "pt": "nitrogen_pt",
                "saturation": "nitrogen_saturation",
            },
            "Oxygen": "oxygen_pt",
            "n-Dodecane": "ndodecane_pt",
        },
    )

    pt = source.state_pt("Nitrogen", 30.0e6, 300.0)
    assert pt.P == pytest.approx(30.0e6)
    assert pt.T == pytest.approx(300.0)
    assert pt.rho > 0.0
    assert pt.gamma > 0.0
    saturation = source.saturation_at_p("Nitrogen", 500_000.0)
    assert saturation.liquid.rho > saturation.vapor.rho
    assert source.saturation_bounds("Nitrogen")[0] < 500_000.0
