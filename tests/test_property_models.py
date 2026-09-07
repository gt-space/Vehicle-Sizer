from dataclasses import dataclass
from pathlib import Path

import pytest

from Flight.FluidNetwork import FluidNetwork
from Flight.FluidNode import CombustorComponent, FlowConn
from FluidProperties.PropertyModels import (
    CombustionProperties,
    PureFluidProperties,
    TableCombustionPropertySource,
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

    def state_rho_u(self, fluid, density, internal_energy):
        self.calls += 1
        return PureFluidProperties(
            P=200_000.0,
            T=300.0,
            rho=density,
            h=internal_energy + 200_000.0 / density,
            u=internal_energy,
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
                "state0": {"m": 2.0, "U": 600_000.0},
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
        FlowConn(
            "ox",
            1.0,
            {
                "components": {"ox": {"phase": "liquid", "mdot": 2.0}},
                "enabled": True,
            },
        ),
        FlowConn(
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
