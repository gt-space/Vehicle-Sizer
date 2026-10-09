"""Generic CLI tables from the built design and synchronized flight states."""
from copy import deepcopy
from collections import Counter
import math

import numpy as np

PSI = 6894.757293168
G0 = 9.80665


def collect_design_summary(cfg, vehicle, propulsion):
    """Capture the dry assembled vehicle before fluid inventories are attached."""
    sections, tanks = [], []
    section_details, tank_details = {}, {}
    counts = Counter()
    names = {"nosecone": "Nosecone", "avi_bay": "Avionics bay",
             "fin_can": "Fin can / boattail", "inter_tank": "Intertank"}
    for definition, section in zip(cfg["vehicle"]["sections"], vehicle.sections):
        kind = definition["type"]
        counts[kind] += 1
        label = definition.get("name", definition.get("tank_id", names.get(kind, kind)))
        if kind == "inter_tank" and "name" not in definition:
            label = f"Intertank {counts[kind]}"
        section_mass = float(np.sum(section.mass))
        section_key = (definition.get("tank_id") or
                       f"{kind}_{counts[kind]}")
        section_details[section_key] = dict(
            station=float(section.start_station), length=float(section.length),
            mass=section_mass)
        sections.append([label, float(section.start_station), float(section.length),
                         section_mass])
        if kind == "fin_can":
            sections.append(["Engine", float(vehicle.engine_start_station),
                             float(vehicle.engine.length), float(vehicle.engine.mass)])
            section_details['engine'] = dict(
                station=float(vehicle.engine_start_station),
                length=float(vehicle.engine.length),
                mass=float(vehicle.engine.mass))
        if "tank_id" in definition:
            tank_id = definition["tank_id"]
            tank_cfg = cfg["tanks"][tank_id]
            dry_mass = float(np.sum(section.dry_mass))
            shell_mass = float(np.sum(section.shell_mass))
            tank_details[tank_id] = dict(
                mass=dry_mass, shell_mass=shell_mass,
                hardware_mass=dry_mass - shell_mass,
                volume=float(section.volume), length=float(section.length),
                diameter=float(section.diameter if tank_cfg["type"] == "pressurant" else section.OMLD),
                thickness=float(section.wall_thickness))
            tanks.append(dict(
                id=tank_id, type=tank_cfg["type"],
                mass=dry_mass, shell_mass=shell_mass,
                hardware_mass=dry_mass - shell_mass,
                volume=float(section.volume), length=float(section.length),
                diameter=float(section.diameter if tank_cfg["type"] == "pressurant" else section.OMLD),
                thickness=float(section.wall_thickness),
                material=(f"Equivalent density {tank_cfg['equivalent_density']:g} kg/m³"
                          if tank_cfg["type"] == "pressurant" and tank_cfg.get("construction", "metal") == "copv" and "equivalent_density" in tank_cfg
                          else tank_cfg.get("material", "N/A")),
                initial=propulsion.initial_states[tank_id],
            ))
    sl = propulsion.combustion_properties.evaluate(
        chamber_pressure=propulsion.Pc_target, mixture_ratio=propulsion.MR_target,
        ambient_pressure=101325., expansion_ratio=propulsion.expansion_ratio,
        cstar_efficiency=propulsion.cstar_efficiency, cf_efficiency=propulsion.cf_efficiency,
    )
    physical_exit_diameter = 2 * np.sqrt(propulsion.throat_area * propulsion.expansion_ratio / np.pi)
    return dict(
        battery_sizing=deepcopy(getattr(vehicle, "battery_sizing", {})),
        sections=sections, tanks=tanks,
        section=section_details, tank={**tank_details,
            'total_mass': sum(t['mass'] for t in tank_details.values()),
            'total_shell_mass': sum(t['shell_mass'] for t in tank_details.values()),
            'total_hardware_mass': sum(t['hardware_mass'] for t in tank_details.values()),
        },
        length=float(vehicle.length),
        diameter=float(cfg["vehicle"]["OMLD"]),
        dry_inertia={"Ixx": float(vehicle.Ixx), "Iyy": float(vehicle.Iyy)},
        design_mdot=float(propulsion.mdot_total),
        design_thrust=float(propulsion.design_thrust),
        throat_area=float(propulsion.throat_area),
        expansion_ratio=float(propulsion.expansion_ratio),
        physical_exit_diameter=float(physical_exit_diameter),
        aero_exit_diameter=float(cfg["engine"].get("exit_diameter", physical_exit_diameter)),
        sea_level_thrust=float(sl.Cf * propulsion.Pc_target * propulsion.throat_area),
    )


def _number(value, digits=2):
    return f"{value:,.{digits}f}" if value is not None and math.isfinite(value) else "N/A"


def _table(title, fields, rows):
    from prettytable import PrettyTable
    table = PrettyTable(fields)
    table.title = title
    table.align = "r"
    table.align[fields[0]] = "l"
    table.add_rows(rows)
    return str(table)


def build_run_tables(cfg, result):
    """Return tables in display order; never infer launch values from the first step."""
    design, initial = result.design_summary, result.initial_state
    if not design or initial is None:
        return []
    history = result.history or []
    states = [initial, *history]
    shutdown = next((s for s in states if s["plant"].fluids.propulsion.mode == "shutdown"), None)
    final = states[-1]
    nodes = initial["plant"].fluids.node
    tank_nodes = {node["tank_id"]: node for node in nodes.values() if node.get("tank_id")}
    tables = []
    battery = result.battery_sizing
    if battery:
        tables.append(_table("Electric pump battery (power-limited)", ["Parameter", "Value"], [
            ["Pump electrical power [W]", _number(battery["power_draw_w"])],
            ["Cells (series x parallel)", f'{battery["cells_in_series"]} x {battery["cells_in_parallel"]}'],
            ["Calculated mass [kg]", _number(battery["calculated_mass_kg"], 3)],
            ["Selected mass [kg]", _number(battery["selected_mass_kg"], 3)],
            ["Separate battery mass used [kg]", _number(battery["used_mass_kg"], 3)],
            ["Section (nose to aft)", battery["section_index"] + 1],
            ["Mass source", battery["mass_source"]],
            ["Total section override [kg]", _number(battery["section_mass_override_kg"], 3)],
        ]))
    tables.append(_table("Vehicle sections (nose to aft; engine overlaps its housing)",
                         ["Section", "Station [m]", "Length [m]", "Dry mass [kg]"],
                         [[name, _number(x, 3), _number(length, 4), _number(mass)]
                          for name, x, length, mass in design["sections"]]))
    tables.append(_table("Overall vehicle", ["Parameter", "Value"], [
        ["Length", f'{design["length"]:.3f} m / {design["length"] / .3048:.2f} ft'],
        ["Diameter", f'{design["diameter"] * 1000:.2f} mm / {design["diameter"] / .0254:.2f} in'],
    ]))
    tank_columns = []
    for tank in design["tanks"]:
        node = tank_nodes[tank["id"]]
        load = node.get("m_liq") if tank["type"] == "propellant" else node.get("mass", node.get("m"))
        tank_columns.append([
            _number(load, 3), _number(tank["volume"] * 1000),
            _number(tank["diameter"] * 1000), _number(tank["length"], 3),
            _number(tank["thickness"] * 1000, 3), tank["material"],
            _number(node.get("T_liq", node.get("T", tank["initial"]["T"]))),
        ])
    if tank_columns:
        labels = ["Initial fluid load [kg]", "Volume [L]", "Outer diameter [mm]",
                  "Length [m]", "Wall thickness [mm]", "Material", "Initial temperature [K]"]
        tables.append(_table("Tanks (propellant load excludes ullage gas)",
                             ["Parameter", *[t["id"] for t in design["tanks"]]],
                             [[label, *values] for label, values in zip(labels, zip(*tank_columns))]))
    prop_mass = sum(tank_nodes[t["id"]].get("m_liq", 0.) for t in design["tanks"] if t["type"] == "propellant")
    stored_gas = result.initial_mass - result.dry_mass - prop_mass
    tables.append(_table("Vehicle mass", ["Component / state", "Mass [kg]"], [
        ["Dry vehicle", _number(result.dry_mass)], ["Initial propellant", _number(prop_mass)],
        ["Initial pressurant / other stored fluids", _number(stored_gas)],
        ["Launch mass", _number(result.initial_mass)],
        ["Mass after shutdown", _number(shutdown["mass_properties"]["total_mass"] if shutdown else None)],
        ["Final mass", _number(final["mass_properties"]["total_mass"])],
    ]))
    pressure_rows = [[node_id, _number(node["P"] / PSI)] for node_id, node in nodes.items() if "P" in node]
    if "exit_pressure" in cfg["engine"]:
        label = "Nozzle exit pressure input" if "expansion_ratio" in cfg["engine"] else "Nozzle exit (design)"
        pressure_rows.append([label, _number(cfg["engine"]["exit_pressure"] / PSI)])
    tables.append(_table("Pressure ladder at initialization (absolute)", ["Node / location", "Pressure [psia]"], pressure_rows))
    max_speed = max(math.hypot(s["kinematics"].vx, s["kinematics"].vz) for s in states)
    max_mach = max(s["atmosphere"].Ma for s in states)
    # Proper acceleration excludes gravity, unlike vertical acceleration dvz/dt.
    max_g = max(math.hypot(s["forces"]["thrust"] - s["plant"].aero.A,
                          s["plant"].aero.N) / s["mass_properties"]["total_mass"] / G0 for s in states)
    rail = next((s for s in states if result.rail_exit_time is not None
                 and s["kinematics"].t >= result.rail_exit_time - 1e-6), None)
    tables.append(_table("Kinematics (initial and accepted samples)", ["Parameter", "Value"], [
        ["Off-rail TWR", _number(rail["forces"]["twr"] if rail else None)],
        ["Maximum ground speed", f"{max_speed:,.2f} m/s"],
        ["Maximum Mach", _number(max_mach, 3)],
        ["Maximum proper acceleration", f"{max_g:.3f} g"],
        ["Maximum Q", f"{result.max_q / 1000:.2f} kPa / {result.max_q / PSI:.2f} psi"],
        ["Apogee", f"{result.apogee / 1000:.3f} km" if result.apogee_reached else "Not reached"],
        ["Termination", result.termination],
    ]))
    engine, prop = cfg["engine"], cfg["prop_system"]
    thrust = float(design["design_thrust"])
    max_thrust = max(s["plant"].fluids.propulsion.thrust for s in states)
    tables.append(_table("Engine", ["Parameter", "Value"], [
        ["Design thrust", f"{thrust / 1000:.2f} kN / {thrust / 4.4482216152605:,.0f} lbf"],
        ["Throat area [m²]", _number(design["throat_area"], 8)],
        ["Expansion ratio", _number(design["expansion_ratio"], 4)],
        ["Physical nozzle exit diameter [m]", _number(design["physical_exit_diameter"], 6)],
        ["Aero exit diameter [m]", _number(design["aero_exit_diameter"], 6)],
        ["Sea-level thrust (design Pc/MR)", f'{design["sea_level_thrust"] / 1000:.2f} kN'],
        ["Initial launch thrust", f'{initial["plant"].fluids.propulsion.thrust / 1000:.2f} kN'],
        ["Maximum sampled thrust", f"{max_thrust / 1000:.2f} kN"],
        ["Design mixture ratio O/F", _number(prop["MR_target"], 3)],
        ["Design propellant flow", f'{design["design_mdot"]:.3f} kg/s'],
        ["Burn duration", f'{result.burn_duration:.3f} s' + (" (incomplete)" if not result.burn_complete else "")],
        ["Eta Cf", _number(engine["cf_efficiency"], 3)],
        ["Eta C*", _number(engine["cstar_efficiency"], 3)],
    ]))
    inertia_rows = []
    for label, properties in (
        ("Dry", design["dry_inertia"]), ("Launch", initial["mass_properties"]),
        ("Shutdown", shutdown["mass_properties"] if shutdown else None),
        ("Final", final["mass_properties"]),
    ):
        inertia_rows.append([label, *[_number(properties[key], 3) if properties else "N/A"
                                      for key in ("Ixx", "Iyy", "Iyy")]])
    tables.append(_table("Body-axis inertia about each state's CG [kg m^2]",
                         ["State", "Ixx (longitudinal)", "Iyy (pitch)", "Izz (assumed = Iyy)"], inertia_rows))
    return tables


def print_run_summary(cfg, result):
    for table in build_run_tables(cfg, result):
        print(table)
        print()
