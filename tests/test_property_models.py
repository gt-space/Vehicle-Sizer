from dataclasses import dataclass
from pathlib import Path

import pytest

from Fluids.FluidNode import CombustorComponent, VolumeComponent
from Fluids.FluidState import BranchState, FluidState
from FluidTables.PropertyModels import (
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

    @staticmethod
    def supports_saturation(fluid):
        return False

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
            mu=1.8e-5,
            k=0.026,
            cp=1000.0,
            beta=1.0 / temperature,
        )


class FixedCombustionProperties:
    def __init__(self):
        self.calls = 0
        self.conditions = None

    def evaluate(self, **conditions):
        self.calls += 1
        self.conditions = conditions
        return CombustionProperties(
            cstar=1500.0,
            Cf=1.5,
            R=350.0,
            gamma=1.2,
            T=3000.0,
        )


def test_gas_node_accepts_an_injected_property_source():
    properties = FixedFluidProperties()
    node = VolumeComponent("tank", {
        "fluid": "test_gas", "geometry": GasGeometry(volume=1.),
        "state0": {"P": 2e5, "T": 300., "m": 2., "U": 6e5},
    }, fluid_properties=properties)
    result = node.evaluate(node.initial_values())
    assert properties.calls > 0
    assert result["P"] == 2e5
    assert result.fluids["test_gas"]["rho"] == 2.


@pytest.mark.parametrize(
    ("oxidizer_mdot", "fuel_mdot", "expected_mr"),
    ((2.0, 1.0, 2.0), (20.0, 1.0, 10.0), (0.05, 1.0, 0.1)),
)
def test_combustion_node_accepts_an_injected_property_source(
    oxidizer_mdot, fuel_mdot, expected_mr
):
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
        }, combustion_properties=properties,
    )
    adjacent = [
        branch_flow(
            "ox",
            1.0,
            {
                "components": {"ox": {"phase": "liquid", "mdot": oxidizer_mdot}},
                "enabled": True,
            },
        ),
        branch_flow(
            "fuel",
            1.0,
            {
                "components": {"fuel": {"phase": "liquid", "mdot": fuel_mdot}},
                "enabled": True,
            },
        ),
    ]

    result = node.evaluate(
        {"P": 2.0e6}, adjacent, {"ambient": {"P": 100_000.0}}
    )

    assert properties.calls == 1
    assert result["MR"] == pytest.approx(expected_mr)
    assert properties.conditions["mixture_ratio"] == pytest.approx(expected_mr)
    assert result["cstar"] == 1500.0
    assert result["fluids"]["products"]["T"] == 3000.0


def test_table_combustion_source_sizes_and_evaluates_engine():
    root = Path(__file__).resolve().parents[1]
    source = TableCombustionPropertySource(
        root / "FluidTables" / "sizer_lookups.h5", nfz=1
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
    with pytest.raises(ValueError, match="mixture_ratio"):
        source.evaluate(
            chamber_pressure=2.0e6,
            mixture_ratio=0.05,
            ambient_pressure=101_325.0,
            expansion_ratio=expansion_ratio,
        )
    assert properties.T > 0.0


def test_table_fluid_source_evaluates_pt_without_inversion():
    root = Path(__file__).resolve().parents[1]
    source = TablePureFluidPropertySource(
        root / "FluidTables" / "sizer_lookups.h5",
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
    assert min(pt.mu, pt.k, pt.cp, pt.beta) > 0.0
    saturation = source.saturation_at_p("Nitrogen", 500_000.0)
    assert saturation.liquid.rho > saturation.vapor.rho
    assert min(saturation.liquid.mu, saturation.vapor.mu) > 0.0
    assert source.saturation_bounds("Nitrogen")[0] < 500_000.0


def test_table_cache_exact_keys_and_derivative_ownership():
    import numpy as np
    source = TablePureFluidPropertySource(
        Path(__file__).resolve().parents[1] / 'FluidTables/sizer_lookups.h5',
        {'Nitrogen': 'nitrogen_pt'})
    state = source.state_pt('Nitrogen', 3e6, 300.)
    assert source.state_pt('Nitrogen', 3e6, 300.) is state
    different = source.state_pt('Nitrogen', np.nextafter(3e6, np.inf), 300.)
    assert different.P != state.P  # No coordinate rounding, even one ULP apart.
    slopes = source.derivatives_pt('Nitrogen', 3e6, 300.)
    expected = slopes.copy()
    slopes['drho_dP'] = -123.
    assert source.derivatives_pt('Nitrogen', 3e6, 300.) == expected
    assert source._cached_state_pt.cache_info().hits == 1
    assert source._cached_derivatives_pt.cache_info().hits == 1
    assert source._cached_state_pt.cache_info().maxsize == 8192
    with pytest.raises(ValueError):
        source.state_pt('Nitrogen', -1., 300.)


def test_state_and_derivatives_share_interpolation_and_keep_outputs_owned(monkeypatch):
    source = TablePureFluidPropertySource(
        Path(__file__).resolve().parents[1] / 'FluidTables/sizer_lookups.h5',
        {'Nitrogen': 'nitrogen_pt'})
    table = source._table('Nitrogen')
    original, queries = table.evaluate, []
    def evaluate(*args, **kwargs):
        queries.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(table, 'evaluate', evaluate)
    source.derivatives_pt('Nitrogen', 3e6, 300.)
    state = source.state_pt('Nitrogen', 3e6, 300.)
    assert len(queries) == 1
    exported = state.as_dict()
    exported['P'] = -1.
    assert state.P == 3e6
