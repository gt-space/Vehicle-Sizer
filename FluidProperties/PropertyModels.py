from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol

from scipy.optimize import root_scalar

from Flight.FluidsDef import FluidsDef
from .LookupTables import LookupTables


@dataclass(frozen=True)
class PureFluidProperties:
    """Thermodynamic properties required by fluid nodes and branches."""

    P: float
    T: float
    rho: float
    h: float
    u: float
    R: float
    gamma: float

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> "PureFluidProperties":
        return cls(**{name: float(values[name]) for name in cls.__annotations__})

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SaturationProperties:
    """Coexisting liquid and vapor states for one pure fluid."""

    P: float
    T: float
    liquid: PureFluidProperties
    vapor: PureFluidProperties


@dataclass(frozen=True)
class CombustionProperties:
    """Combustion and nozzle properties required by the propulsion model."""

    cstar: float
    Cf: float
    R: float
    gamma: float
    T: float

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> "CombustionProperties":
        return cls(**{name: float(values[name]) for name in cls.__annotations__})

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


class PureFluidPropertySource(Protocol):
    """Direct pressure-temperature property interface."""

    def state_pt(
        self, fluid: str, pressure: float, temperature: float
    ) -> PureFluidProperties: ...

    def saturation_bounds(self, fluid: str) -> tuple[float, float]: ...

    def supports_saturation(self, fluid: str) -> bool: ...

    def saturation_at_p(
        self, fluid: str, pressure: float
    ) -> SaturationProperties: ...


class CombustionPropertySource(Protocol):
    """Interface shared by CEA and future tabular combustion properties."""

    def expansion_ratio(
        self, chamber_pressure: float, mixture_ratio: float, exit_pressure: float
    ) -> float: ...

    def evaluate(
        self,
        chamber_pressure: float,
        mixture_ratio: float,
        ambient_pressure: float,
        expansion_ratio: float,
        cstar_efficiency: float = 1.0,
        cf_efficiency: float = 1.0,
    ) -> CombustionProperties: ...


class CoolPropPropertySource:
    """Fluid-property source backed by CoolProp."""

    @staticmethod
    def state_pt(
        fluid: str, pressure: float, temperature: float
    ) -> PureFluidProperties:
        return PureFluidProperties.from_dict(
            FluidsDef.coolprop_state(fluid, "P", pressure, "T", temperature)
        )

    @staticmethod
    def saturation_bounds(fluid: str) -> tuple[float, float]:
        from CoolProp.CoolProp import PropsSI

        return float(PropsSI("PTRIPLE", fluid)), float(PropsSI("PCRIT", fluid))

    @staticmethod
    def supports_saturation(fluid: str) -> bool:
        return True

    @staticmethod
    def saturation_at_p(fluid: str, pressure: float) -> SaturationProperties:
        liquid = PureFluidProperties.from_dict(
            FluidsDef.coolprop_state(fluid, "P", pressure, "Q", 0.0)
        )
        vapor = PureFluidProperties.from_dict(
            FluidsDef.coolprop_state(fluid, "P", pressure, "Q", 1.0)
        )
        return SaturationProperties(pressure, vapor.T, liquid, vapor)


class TablePureFluidPropertySource:
    """Pure-fluid properties directly interpolated from pressure-temperature maps."""

    outputs = (
        "density",
        "enthalpy",
        "internal_energy",
        "specific_heat_ratio",
    )

    def __init__(
        self,
        lookup_file: str | Path,
        fluid_tables: Mapping[str, str | Mapping[str, str]],
    ) -> None:
        if not fluid_tables:
            raise ValueError("Table fluid source requires fluid-to-table mappings")
        self.fluid_tables = {
            fluid: table if isinstance(table, str) else table["pt"]
            for fluid, table in fluid_tables.items()
        }
        self.saturation_tables = {
            fluid: table["saturation"]
            for fluid, table in fluid_tables.items()
            if not isinstance(table, str) and "saturation" in table
        }
        table_names = (*self.fluid_tables.values(), *self.saturation_tables.values())
        self.tables = LookupTables(
            lookup_file,
            table_names=tuple(dict.fromkeys(table_names)),
        )

    def _table(self, fluid: str):
        try:
            return self.tables[self.fluid_tables[fluid]]
        except KeyError as error:
            raise KeyError(
                f"No property table configured for fluid {fluid!r}"
            ) from error

    def state_pt(
        self, fluid: str, pressure: float, temperature: float
    ) -> PureFluidProperties:
        table = self._table(fluid)
        if table.axis_order != ("pressure", "temperature"):
            raise ValueError(
                f"Fluid table '{table.name}' must use pressure and temperature axes"
            )
        missing = set(self.outputs).difference(table.outputs)
        if missing:
            raise ValueError(
                f"Fluid table '{table.name}' is missing outputs {sorted(missing)}"
            )
        try:
            gas_constant = float(table.constants["specific_gas_constant"])
        except KeyError as error:
            raise ValueError(
                f"Fluid table '{table.name}' requires specific_gas_constant"
            ) from error
        values = table.evaluate(
            self.outputs,
            pressure=float(pressure),
            temperature=float(temperature),
        )
        if values["density"] <= 0.0 or values["specific_heat_ratio"] <= 0.0:
            raise ValueError(f"Fluid table '{table.name}' returned an invalid state")
        return PureFluidProperties(
            P=float(pressure),
            T=float(temperature),
            rho=values["density"],
            h=values["enthalpy"],
            u=values["internal_energy"],
            R=gas_constant,
            gamma=values["specific_heat_ratio"],
        )

    def _saturation_table(self, fluid: str):
        try:
            return self.tables[self.saturation_tables[fluid]]
        except KeyError as error:
            raise KeyError(
                f"No saturation table configured for fluid {fluid!r}"
            ) from error

    def saturation_bounds(self, fluid: str) -> tuple[float, float]:
        pressure = self._saturation_table(fluid).axes["pressure"]
        return float(pressure[0]), float(pressure[-1])

    def supports_saturation(self, fluid: str) -> bool:
        return fluid in self.saturation_tables

    def saturation_at_p(
        self, fluid: str, pressure: float
    ) -> SaturationProperties:
        table = self._saturation_table(fluid)
        if table.axis_order != ("pressure",):
            raise ValueError(
                f"Saturation table '{table.name}' must use a pressure axis"
            )
        names = (
            "temperature",
            "liquid_density",
            "liquid_enthalpy",
            "liquid_internal_energy",
            "liquid_specific_heat_ratio",
            "vapor_density",
            "vapor_enthalpy",
            "vapor_internal_energy",
            "vapor_specific_heat_ratio",
        )
        values = table.evaluate(names, pressure=float(pressure))
        gas_constant = float(table.constants["specific_gas_constant"])

        def phase(prefix: str) -> PureFluidProperties:
            return PureFluidProperties(
                P=float(pressure),
                T=values["temperature"],
                rho=values[f"{prefix}_density"],
                h=values[f"{prefix}_enthalpy"],
                u=values[f"{prefix}_internal_energy"],
                R=gas_constant,
                gamma=values[f"{prefix}_specific_heat_ratio"],
            )

        return SaturationProperties(
            P=float(pressure),
            T=values["temperature"],
            liquid=phase("liquid"),
            vapor=phase("vapor"),
        )


class CEAPropertySource:
    """Combustion-property source backed by an initialized RocketCEA object."""

    def __init__(self, cea: Any) -> None:
        self.cea = cea

    def expansion_ratio(
        self,
        chamber_pressure: float,
        mixture_ratio: float,
        exit_pressure: float,
    ) -> float:
        if not 0.0 < exit_pressure < chamber_pressure:
            raise ValueError("Exit pressure must be between zero and chamber pressure")
        return float(
            self.cea.get_eps_at_PcOvPe(
                chamber_pressure,
                mixture_ratio,
                chamber_pressure / exit_pressure,
            )
        )

    def evaluate(
        self,
        chamber_pressure: float,
        mixture_ratio: float,
        ambient_pressure: float,
        expansion_ratio: float,
        cstar_efficiency: float = 1.0,
        cf_efficiency: float = 1.0,
    ) -> CombustionProperties:
        return CombustionProperties.from_dict(
            FluidsDef.combustion_properties(
                chamber_pressure=chamber_pressure,
                mixture_ratio=mixture_ratio,
                ambient_pressure=ambient_pressure,
                expansion_ratio=expansion_ratio,
                cea=self.cea,
                cstar_efficiency=cstar_efficiency,
                cf_efficiency=cf_efficiency,
            )
        )


class TableCombustionPropertySource:
    """Combustion-property source backed by ``engine_lookup``."""

    outputs = (
        "chamber_temperature",
        "chamber_gamma",
        "chamber_molecular_weight",
        "characteristic_velocity",
        "thrust_coefficient",
    )

    def __init__(self, lookup_file: str | Path, nfz: int) -> None:
        self.table = LookupTables(
            lookup_file, table_names=("engine_lookup",)
        )["engine_lookup"]
        self.nfz = float(nfz)
        if not any(self.table.axes["nfz"] == self.nfz):
            raise ValueError(
                f"Engine nfz must be one of {self.table.axes['nfz'].tolist()}"
            )
        missing = set(self.outputs).difference(self.table.outputs)
        if missing:
            raise ValueError(f"Engine table is missing outputs {sorted(missing)}")

    def _coordinates(
        self,
        chamber_pressure: float,
        mixture_ratio: float,
        expansion_ratio: float,
        ambient_pressure: float,
    ) -> Dict[str, float]:
        return {
            "chamber_pressure": float(chamber_pressure),
            "mixture_ratio": float(mixture_ratio),
            "expansion_ratio": float(expansion_ratio),
            "ambient_pressure": float(ambient_pressure),
            "nfz": self.nfz,
        }

    def expansion_ratio(
        self,
        chamber_pressure: float,
        mixture_ratio: float,
        exit_pressure: float,
    ) -> float:
        if not 0.0 < exit_pressure < chamber_pressure:
            raise ValueError("Exit pressure must be between zero and chamber pressure")
        epsilon = self.table.axes["expansion_ratio"]

        def pressure_error(expansion_ratio: float) -> float:
            calculated = self.table.get(
                "exit_pressure",
                **self._coordinates(
                    chamber_pressure,
                    mixture_ratio,
                    expansion_ratio,
                    exit_pressure,
                ),
            )
            return calculated - exit_pressure

        lower_error = pressure_error(float(epsilon[0]))
        upper_error = pressure_error(float(epsilon[-1]))
        if lower_error == 0.0:
            return float(epsilon[0])
        if upper_error == 0.0:
            return float(epsilon[-1])
        if lower_error * upper_error > 0.0:
            raise ValueError(
                "Target exit pressure is not bracketed by the engine table's "
                "expansion-ratio range"
            )
        solution = root_scalar(
            pressure_error,
            bracket=(float(epsilon[0]), float(epsilon[-1])),
            method="brentq",
        )
        if not solution.converged:
            raise RuntimeError("Engine-table expansion-ratio solve failed")
        return float(solution.root)

    def evaluate(
        self,
        chamber_pressure: float,
        mixture_ratio: float,
        ambient_pressure: float,
        expansion_ratio: float,
        cstar_efficiency: float = 1.0,
        cf_efficiency: float = 1.0,
    ) -> CombustionProperties:
        values = self.table.evaluate(
            self.outputs,
            **self._coordinates(
                chamber_pressure,
                mixture_ratio,
                expansion_ratio,
                ambient_pressure,
            ),
        )
        molecular_weight = values["chamber_molecular_weight"]
        if molecular_weight <= 0.0:
            raise ValueError("Engine table returned a nonpositive molecular weight")
        return CombustionProperties(
            cstar=values["characteristic_velocity"] * cstar_efficiency,
            Cf=values["thrust_coefficient"] * cf_efficiency,
            R=8314.462618 / molecular_weight,
            gamma=values["chamber_gamma"],
            T=values["chamber_temperature"],
        )
