"""Side-effect-free simulation entry point: no plots, output files or progress by default."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path

from AeroTables import DragModel
from Flight.Flight import FlightSim
from Fluids.PropSystem import PropSystem, DesignInfeasible
from Flight.environment import Environment
from Flight.flight_forces import Aero
from FluidProperties.PropertyModels import (
    CEAPropertySource,
    CoolPropPropertySource,
    TableCombustionPropertySource,
    TablePureFluidPropertySource,
)
from Vehicle.Engine import Engine
from Vehicle.Vehicle import Vehicle
from Thermals import ThermalNetwork
from simulation_types import SimResult
from constraints import (GeometryError, EvaluationFailure, OperatingInfeasible,
                         configured_limits, finalize, merge_margins)

ROOT = Path(__file__).resolve().parent


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def property_sources(cfg: dict):
    """Construct the two property interfaces selected by the config."""

    pure_cfg = cfg["property_models"]["pure_fluid"]
    if pure_cfg["source"] == "coolprop":
        pure = CoolPropPropertySource()
    elif pure_cfg["source"] == "table":
        table_cfg = pure_cfg["table"]
        pure = TablePureFluidPropertySource(
            project_path(table_cfg["lookup_file"]),
            table_cfg["fluids"],
        )
    else:
        raise ValueError(f"Unknown pure-fluid source {pure_cfg['source']!r}")

    combustion_cfg = cfg["property_models"]["combustion"]
    if combustion_cfg["source"] == "cea":
        from rocketcea.cea_obj_w_units import CEA_Obj

        engine = cfg["engine"]
        cea = CEA_Obj(
            oxName=engine["oxidizer"],
            fuelName=engine["fuel"],
            pressure_units="Pa",
            cstar_units="m/s",
            temperature_units="K",
            enthalpy_units="J/kg",
            density_units="kg/m^3",
            specific_heat_units="J/kg-K",
        )
        combustion = CEAPropertySource(cea)
    elif combustion_cfg["source"] == "table":
        table_cfg = combustion_cfg["table"]
        combustion = TableCombustionPropertySource(
            project_path(table_cfg["lookup_file"]),
            int(table_cfg["nfz"]),
        )
    else:
        raise ValueError(
            f"Unknown combustion source {combustion_cfg['source']!r}"
        )
    return pure, combustion


def simulate(cfg: dict, *, pure_properties=None, combustion_properties=None,
             aero_model=None, record_history=False, compute_loads=False,
             progress=None) -> SimResult:
    """Evaluate a fresh candidate; expected rejects return, unexpected failures raise.

    EvaluationFailure deliberately escapes to stop an optimizer. Its cause and
    copied config allow reproduction without printing or writing files here.
    """
    context = {"phase": "configuration", "flight": None}
    candidate = deepcopy(cfg)
    try:
        return _simulate(candidate, pure_properties=pure_properties,
                         combustion_properties=combustion_properties,
                         aero_model=aero_model, record_history=record_history,
                         compute_loads=compute_loads, progress=progress, context=context)
    except Exception as error:
        raise EvaluationFailure(context["phase"], candidate,
                                getattr(context["flight"], "result", None)) from error


def _simulate(cfg, *, pure_properties, combustion_properties, aero_model,
              record_history, compute_loads, progress, context):
    """Build fresh mutable state and run one candidate without output side effects.

    Reuse injected read-only property sources/aero_model across candidates to
    avoid reloading tables. Expected construction and converged operating-limit
    rejections become partial results; unresolved solver errors remain fatal.
    """
    limits = configured_limits(cfg)
    if (pure_properties is None) != (combustion_properties is None):
        raise ValueError("Inject both property sources, or neither")
    if pure_properties is None:
        pure_properties, combustion_properties = property_sources(cfg)
    context["phase"] = "sizing"
    propulsion = None
    try:
        vehicle = Vehicle(cfg, pure_properties)
        propulsion = PropSystem(cfg, vehicle.tanks, fluid_properties=pure_properties,
                                combustion_properties=combustion_properties)
        vehicle.build(Engine(float(cfg["engine"]["mass"]), float(cfg["engine"]["length"]), propulsion.exit_area))
    except DesignInfeasible as error:
        geometry = isinstance(error, GeometryError)
        result = SimResult(termination="infeasible_initial_design",
                         constraints=dict(getattr(propulsion, "sizing_constraints", {})) if geometry else dict(error.constraints),
                         geometry_constraints=dict(error.constraints) if geometry else {},
                         pump_sizing=getattr(propulsion, "pump_sizing", error.pump_sizing),
                         max_altitude=float(cfg["launch"]["altitude"]),
                         final_altitude=float(cfg["launch"]["altitude"]),
                         final_velocity=float(cfg["launch"]["velocity"]))
        return finalize(result, limits)
    context["phase"] = "flight initialization"
    heating = cfg.get("thermal", {}).get("external_heating", False)
    if not isinstance(heating, bool):
        raise ValueError("thermal.external_heating must be true or false")
    thermal = ThermalNetwork(cfg, vehicle) if heating else None
    environment = Environment(h_max=float(cfg["environment"]["max_altitude"]),
                              dh=float(cfg["environment"]["altitude_step"]))
    if aero_model is None:
        aero_model = DragModel(project_path(cfg["aero"]["model"]))
    flight = FlightSim(cfg, environment, Aero(cfg["aero"], vehicle.aero_candidate(), aero_model),
                       propulsion, vehicle, thermal=thermal)
    context.update(phase="runtime", flight=flight)
    try:
        flight.run(h0=float(cfg["launch"]["altitude"]), v0=float(cfg["launch"]["velocity"]),
                   progress=progress, record_history=record_history, compute_loads=compute_loads)
    except OperatingInfeasible as error:
        result = getattr(flight, "result", None) or SimResult(
            initial_mass=float(vehicle.total_mass),
            max_altitude=float(cfg["launch"]["altitude"]),
            final_altitude=float(cfg["launch"]["altitude"]),
            final_velocity=float(cfg["launch"]["velocity"]),
            geometry_constraints=dict(vehicle.geometry_constraints),
        )
        result.termination = "infeasible_operating_state"
        merge_margins(result.constraints, error.constraints,
                      times=result.constraint_times, time=error.time)
    else:
        result = flight.result
    result.pump_sizing = deepcopy(propulsion.pump_sizing)
    result.constraints.update(propulsion.sizing_constraints)
    context["phase"] = "constraint reporting"
    return finalize(result, limits)
