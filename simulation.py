"""Side-effect-free simulation entry point: no plots, output files or progress by default."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path

from AeroTables import DragModel
from Flight.Flight import FlightSim
from Fluids.PropSystem import PropSystem
from Fluids.design import prepare_design_inputs
from Flight.environment import Environment
from Flight.flight_forces import Aero
from FluidTables.PropertyModels import property_sources
from Vehicle.Vehicle import Vehicle
from Thermals import ThermalNetwork
from simulation_types import SimResult
from diagnostics.warnings import collect_warnings
from diagnostics.errors import EvaluationFailure, SimulationIncompleteError
from diagnostics.constraints import prepare_assessment, finalize


ROOT = Path(__file__).resolve().parent


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def simulate(cfg: dict, *, pure_properties=None, combustion_properties=None,
             aero_model=None, record_history=False, compute_loads=False,
             progress=None, assess_constraints=False,
             record_design_summary=False) -> SimResult:
    """Run the same physics for all callers; return a completed flight or raise.

    Only optimizers opt into constraint assessment. Design margins cannot stop
    integration; model-domain and numerical failures always use EvaluationFailure.
    Its diagnostic includes the cause type, phase, warnings and partial state.
    """
    context = {"phase": "configuration", "flight": None, "propulsion": None}
    candidate = deepcopy(cfg)
    with collect_warnings() as cautions:
        result = None
        try:
            if type(assess_constraints) is not bool:
                raise TypeError('assess_constraints must be a boolean')
            result = _simulate(candidate, pure_properties=pure_properties,
                               combustion_properties=combustion_properties,
                               aero_model=aero_model, record_history=record_history,
                               compute_loads=compute_loads, progress=progress, context=context,
                               assess_constraints=assess_constraints,
                               record_design_summary=record_design_summary)
            if not result.completed:
                context['phase'] = 'runtime'
                raise SimulationIncompleteError(result)
            return result
        except Exception as error:
            result = result or getattr(context["flight"], "result", None)
            raise EvaluationFailure(context["phase"], candidate, result, warnings=cautions) from error
        finally:
            propulsion = context["propulsion"]
            try:
                if result is not None:
                    result.warnings = cautions
                    result.battery_sizing = context.get("battery_sizing", {})
                    result.design_summary = context.get("design_summary", {})
                    for name in ("pump_sizing", "gg_sizing"):
                        setattr(result, name, deepcopy(getattr(propulsion, name, {})))
                    if propulsion is not None:
                        result.fluid_retries = deepcopy(propulsion.network.retry_diagnostics)
                        result.pressure_tracking = deepcopy(propulsion.network.pressure_tracking.records)
            finally:
                if propulsion is not None:
                    propulsion.close()


def _simulate(cfg, *, pure_properties, combustion_properties, aero_model,
              record_history, compute_loads, progress, context, assess_constraints,
              record_design_summary=False):
    """Build once, propagate once, then optionally assess optimizer limits."""
    if (pure_properties is None) != (combustion_properties is None):
        raise ValueError("Inject both property sources, or neither")
    if pure_properties is None:
        pure_properties, combustion_properties = property_sources(cfg)
    context["phase"] = "sizing"
    template, states, wall_pressures = prepare_design_inputs(cfg)
    vehicle = Vehicle(cfg, pure_properties, initial_states=states, wall_pressures=wall_pressures)
    propulsion = PropSystem(cfg, vehicle.tanks, fluid_properties=pure_properties,
                            combustion_properties=combustion_properties,
                            template=template, initial_states=states)
    context["propulsion"] = propulsion
    vehicle.build(propulsion)
    context["battery_sizing"] = deepcopy(vehicle.battery_sizing)
    limits = prepare_assessment(cfg, vehicle, propulsion) if assess_constraints else {}
    context["phase"] = "flight initialization"
    if record_history or record_design_summary:
        from reporting.run_report import collect_design_summary
        context["design_summary"] = collect_design_summary(cfg, vehicle, propulsion)
    selections = {node.get('tank_id', key): node.get('thermal', {})
                  for key, node in propulsion.node_definitions.items() if key in propulsion.aero_nodes}
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
    flight.run(h0=float(cfg["launch"]["altitude"]), v0=float(cfg["launch"]["velocity"]),
               progress=progress, record_history=record_history, compute_loads=compute_loads)
    result = flight.result
    if assess_constraints:
        context["phase"] = "constraint assessment"
        finalize(result, limits)
    return result
