from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Protocol

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
    """Interface shared by CoolProp and future tabular fluid properties."""

    def state_pt(
        self, fluid: str, pressure: float, temperature: float
    ) -> PureFluidProperties: ...

    def state_pu(
        self,
        fluid: str,
        pressure: float,
        internal_energy: float,
        phase: str,
    ) -> PureFluidProperties: ...

    def state_rho_u(
        self,
        fluid: str,
        density: float,
        internal_energy: float,
    ) -> PureFluidProperties: ...


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
    def state_pu(
        fluid: str,
        pressure: float,
        internal_energy: float,
        phase: str,
    ) -> PureFluidProperties:
        return PureFluidProperties.from_dict(
            FluidsDef.coolprop_state_pu(
                fluid, pressure, internal_energy, phase
            )
        )

    @staticmethod
    def state_rho_u(
        fluid: str,
        density: float,
        internal_energy: float,
    ) -> PureFluidProperties:
        return PureFluidProperties.from_dict(
            FluidsDef.coolprop_state(
                fluid, "Dmass", density, "Umass", internal_energy
            )
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
