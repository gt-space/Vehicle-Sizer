"""Side-effect-free simulation entry point: no plots, output files or progress by default."""
from __future__ import annotations
from copy import deepcopy
from math import pi
from pathlib import Path

from AeroTables import DragModel
from Flight.Flight import FlightSim
from Fluids.PropSystem import PropSystem, DesignInfeasible
from Flight.environment import Environment
from Flight.flight_forces import Aero
from FluidTables.PropertyModels import (
    CEAPropertySource,
    CoolPropPropertySource,
    TableCombustionPropertySource,
    TablePureFluidPropertySource,
)
from Vehicle.Engine import Engine
from Vehicle.Vehicle import Vehicle
from Thermals import ThermalNetwork
from simulation_types import SimResult
from warning import collect_warnings
from constraints import vehicle_limit_margins
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

    EvaluationFailure retains a typed cause and copied configuration. The caller
    decides whether the failure ends this candidate or the whole search.
    """
    context = {"phase": "configuration", "flight": None, "propulsion": None}
    candidate = deepcopy(cfg)
    with collect_warnings() as cautions:
        result = None
        try:
            result = _simulate(candidate, pure_properties=pure_properties,
                               combustion_properties=combustion_properties,
                               aero_model=aero_model, record_history=record_history,
                               compute_loads=compute_loads, progress=progress, context=context)
            return result
        except Exception as error:
            result = getattr(context["flight"], "result", None)
            raise EvaluationFailure(context["phase"], candidate, result) from error
        finally:
            try:
                propulsion = context["propulsion"]
                if propulsion is not None:
                    propulsion.close()
            finally:
                if result is not None:
                    result.warnings = cautions


def _simulate(cfg, *, pure_properties, combustion_properties, aero_model,
              record_history, compute_loads, progress, context):
    """Build fresh mutable state and run one candidate without output side effects.

    Reuse injected read-only property sources/aero_model across candidates to
    avoid reloading tables. Expected construction and converged operating-limit
    rejections become partial results; unresolved solver errors escape with context.
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
        context["propulsion"] = propulsion
        vehicle.build(Engine(float(cfg["engine"]["mass"]), float(cfg["engine"]["length"]), propulsion.exit_area))
        static_margins = vehicle_limit_margins(cfg, vehicle)
    except DesignInfeasible as error:
        geometry = isinstance(error, GeometryError)
        result = SimResult(termination="infeasible_initial_design",
                         constraints=dict(getattr(propulsion, "sizing_constraints", {})) if geometry else dict(error.constraints),
                         geometry_constraints=dict(error.constraints) if geometry else {},
                         pump_sizing=getattr(propulsion, "pump_sizing", error.pump_sizing),
                         max_altitude=float(cfg["launch"]["altitude"]),
                         final_altitude=float(cfg["launch"]["altitude"]),
                         final_velocity=float(cfg["launch"]["velocity"]),
                         final_x=0.0,
                         final_vx=0.0,
                         final_vz=float(cfg["launch"]["velocity"]),
                         final_speed=abs(float(cfg["launch"]["velocity"])),
                         final_pitch_angle=pi / 2,
                         final_pitch_rate=0.0)
        return finalize(result, limits)
    context["phase"] = "flight initialization"
    design_summary = {}
    if record_history:
        from run_report import collect_design_summary
        design_summary = collect_design_summary(cfg, vehicle, propulsion)
    from Thermals.heat_sources import thermal_model
    if {'external_heating', 'nodes'} & cfg.get('thermal', {}).keys():
        raise ValueError('Select thermal.model per dynamic model instead of legacy global heating flags')
    selections = {node.get('tank_id', key): node.get('thermal', {})
                  for key, node in propulsion.node_definitions.items()
                  if thermal_model(node.get('thermal')) == 'Aeroheating'}
    thermal = ThermalNetwork(cfg, vehicle, selections=selections) if selections else None
    environment_cfg = cfg["environment"]
    wind_cfg = environment_cfg.get("wind")
    wind_profile = None
    if wind_cfg is not None:
        if not isinstance(wind_cfg, dict):
            raise ValueError("environment.wind must be a mapping")
        wind_enabled = wind_cfg.get("enabled", False)
        if not isinstance(wind_enabled, bool):
            raise ValueError("environment.wind.enabled must be true or false")
        if wind_enabled:
            profile = wind_cfg.get("profile")
            if not isinstance(profile, str) or not profile.strip():
                raise ValueError("environment.wind.profile is required when wind is enabled")
            wind_profile = project_path(profile)
    environment = Environment(
        h_max=float(environment_cfg["max_altitude"]),
        dh=float(environment_cfg["altitude_step"]),
        wind_profile=wind_profile,
    )
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
            final_x=0.0,
            final_vx=0.0,
            final_vz=float(cfg["launch"]["velocity"]),
            final_speed=abs(float(cfg["launch"]["velocity"])),
            final_pitch_angle=pi / 2,
            final_pitch_rate=0.0,
            geometry_constraints=dict(vehicle.geometry_constraints),
        )
        if getattr(flight, "roll_analysis", None) is not None:
            result.roll = flight.roll_analysis.result
        result.termination = "infeasible_operating_state"
        merge_margins(result.constraints, error.constraints,
                      times=result.constraint_times, time=error.time)
        if "max_aoa_deg" in error.constraints:
            limit = limits.get("max_aoa_deg", 15.0)
            result.max_aoa_deg = max(result.max_aoa_deg, limit - error.constraints["max_aoa_deg"])
    else:
        result = flight.result
    result.pump_sizing = deepcopy(propulsion.pump_sizing)
    result.design_summary = design_summary
    result.constraints.update(static_margins)
    result.pressure_tracking = deepcopy(propulsion.network.pressure_tracking.records)
    result.constraints.update(propulsion.sizing_constraints)
    context["phase"] = "constraint reporting"
    return finalize(result, limits)
