"""Construction clearances [m]: nonnegative is feasible, None is unassessed.

These are packaging checks, not pressure-vessel certification or routing CAD.
"""
import math
from constraints import GeometryError


def check_copv_to_airframe(section, diameter, clearance=0.0):
    return (diameter - 2 * section.wall_thickness - section.copv.diameter) / 2 - clearance


def check_passthrough(tank):
    return (tank.OMLD - 2 * tank.wall_thickness - tank.passthrough_diameter) / 2


def check_feedline(tank, outer_diameter, clearance=0.0):
    bore = tank.passthrough_diameter - 2 * tank.passthrough_wall_thickness
    return (bore - outer_diameter) / 2 - clearance


def check_passthrough_bore(tank):
    return tank.passthrough_diameter / 2 - tank.passthrough_wall_thickness if tank.passthrough_diameter else 0.0


def check_nozzle(fin, engine, clearance=0.0):
    return (fin.boattail_aft_diameter - 2 * fin.wall_thickness - 2 * math.sqrt(engine.exit_area / math.pi)) / 2 - clearance


def check_engine_length(fin, engine):
    return fin.length - engine.length


def check_engine_envelope(fin, engine, diameter, body_diameter, clearance=0.0):
    # Conservative cylindrical engine envelope inside the tapered shell.
    end_diameter = diameter + (fin.boattail_aft_diameter - diameter) * engine.length / fin.length
    return (min(diameter, end_diameter) - 2 * fin.wall_thickness - body_diameter) / 2 - clearance


def geometry_constraints(vehicle):
    from .sections.PressTank import PressTank
    from .sections.PropTank import PropTank
    from .sections.FinCan import FinCan

    cfg = vehicle.cfg
    packaging = cfg.get("geometry", {})
    clearance = float(packaging.get("radial_clearance", 0.0))
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError("geometry.radial_clearance must be finite and nonnegative")
    diameter = float(cfg["vehicle"]["OMLD"])
    margins = {}
    for section in vehicle.sections:
        if isinstance(section, PressTank):
            margins[f"{section.tank_id}.copv_airframe"] = check_copv_to_airframe(section, diameter, clearance)
        elif isinstance(section, PropTank):
            margins[f"{section.tank_id}.passthrough"] = check_passthrough(section) - clearance
            margins[f"{section.tank_id}.passthrough_bore"] = check_passthrough_bore(section)
            margins[f"{section.tank_id}.wall_gauge"] = section.wall_thickness - section.required_wall_thickness
        elif isinstance(section, FinCan):
            margins["engine.length"] = check_engine_length(section, vehicle.engine)
            margins["engine.nozzle"] = check_nozzle(section, vehicle.engine, clearance)
            body = cfg["engine"].get("envelope_diameter")
            if body is not None and (not math.isfinite(float(body)) or float(body) <= 0):
                raise ValueError("Engine envelope_diameter must be finite and positive")
            margins["engine.envelope"] = None if body is None else check_engine_envelope(section, vehicle.engine, diameter, float(body), clearance)
    lines = packaging.get("feedlines", [])
    if not lines:
        margins["feedline.routing"] = None
    for line in lines:
        outer = float(line["outer_diameter"])
        if not math.isfinite(outer) or outer <= 0:
            raise ValueError("Feedline outer_diameter must be finite and positive")
        tank = vehicle.tanks[line["through_tank"]]
        if not isinstance(tank, PropTank):
            raise ValueError("Feedline passthrough checks require a propellant tank")
        key = f"feedline.{line['name']}.{tank.tank_id}"
        if key in margins:
            raise ValueError(f"Duplicate feedline constraint {key}")
        margins[key] = check_feedline(tank, outer, clearance)
    if any(value is not None and not math.isfinite(value) for value in margins.values()):
        raise ValueError("Geometry constraints must be finite")
    return margins
