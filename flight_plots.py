"""Flight-result plots kept separate from simulation assembly."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _series(rows, name, scale=1.0):
    return np.asarray([row[name] for row in rows], dtype=float) * scale


def _burnout_time(rows):
    was_on = False
    for row in rows:
        if row["engine_on"]:
            was_on = True
        elif was_on:
            return float(row["time_s"])
    return None


def _mark_burnout(axes, burnout):
    if burnout is None:
        return
    for axis in np.atleast_1d(axes).flat:
        axis.axvline(burnout, color="black", linestyle="--", linewidth=1, label="Burnout")


def _finish(figure, axes, path, burnout=None):
    _mark_burnout(axes, burnout)
    for axis in np.atleast_1d(axes).flat:
        axis.grid(True, alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            axis.legend(unique.values(), unique.keys(), fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _path(base, name):
    return base if name == "kinematics" else base.with_name(f"{base.stem}_{name}{base.suffix}")


def _load_surface(axis, time, station, values, title, zlabel, burnout):
    """Plot a time history with rocket station on the front axis."""

    station_grid, time_grid = np.meshgrid(station, time)
    limit = max(float(np.nanmax(np.abs(values))), 1.0e-12)
    surface = axis.plot_surface(
        station_grid,
        time_grid,
        values,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        linewidth=0,
        antialiased=False,
        rcount=min(len(time), 100),
        ccount=min(len(station), 200),
    )
    if burnout is not None:
        axis.plot(
            station,
            np.full_like(station, burnout),
            np.zeros_like(station),
            "k--",
            linewidth=1,
            label="Burnout",
        )
    axis.set(
        xlabel="Rocket axial station [m]",
        ylabel="Time [s]",
        zlabel=zlabel,
        title=title,
    )
    axis.view_init(elev=25, azim=-55)
    axis.set_box_aspect((1.6, 1.2, 0.8))
    return surface


def _tank_nodes(history):
    nodes = history[0]["plant"].fluids.node
    return {
        node.get("tank_id"): node_id
        for node_id, node in nodes.items()
        if node.get("tank_id")
    }


def _valve_history(history, branch_id):
    """Reconstruct substep transitions; endpoint samples cannot resolve fast cycling."""
    first = history[0]["plant"].fluids
    key = f"branch:{branch_id}:switch"
    events = [event for state in history for event in state["plant"].fluids.events
              if event["kind"] == "branch" and event["component"] == branch_id
              and "is_open" in event]
    if not events and key not in first.event_counts:
        return ([state["kinematics"].t for state in history],
                [state["plant"].fluids.branch[branch_id]["is_open"] for state in history],
                [0] * len(history))
    kin = history[0]["kinematics"]
    initial_count = events[0]["count"] - 1 if events else first.event_counts[key]
    initial_open = events[0]["was_open"] if events else first.branch[branch_id]["is_open"]
    times = [kin.t - kin.dt] + [event["time_s"] for event in events]
    states = [initial_open] + [event["is_open"] for event in events]
    counts = [initial_count] + [event["count"] for event in events]
    return (times + [history[-1]["kinematics"].t], states + [states[-1]], counts + [counts[-1]])


def plot_valve_actuations(history, path, burnout=None):
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    valves = [key for key, branch in history[0]["plant"].fluids.branch.items() if "is_open" in branch]
    for offset, branch_id in enumerate(valves):
        times, states, counts = _valve_history(history, branch_id)
        label = branch_id.replace("_", " ")
        axes[0].step(times, np.asarray(states, dtype=float) + 1.2 * offset,
                     where="post", label=label)
        axes[1].step(times, counts, where="post", label=f"{label}: {counts[-1]} switches")
    axes[0].set(ylabel="Valve state (offset; 0 closed, 1 open)",
                title="Bang-bang valve actuations")
    axes[1].set(xlabel="Time [s]", ylabel="Cumulative switches (open + close)")
    _finish(figure, axes, path, burnout)


def plot_flight(history: list, rows: list[dict], base_path: Path) -> dict[str, Path]:
    """Write the standard flight diagnostic figures and return their paths."""

    base_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {name: _path(base_path, name) for name in (
        "pressure_ladder", "copv_blowdown", "tank_temperatures",
        "propellant_mass", "pc_thrust", "kinematics", "trajectory", "axial_temperatures",
        "bang_bang", "pressurant_mdot", "engine_mdot_mr", "mass_distribution",
        "axial_force_distribution", "normal_force_distribution",
    )}
    time = _series(rows, "time_s")
    burnout = _burnout_time(rows)
    tank_nodes = _tank_nodes(history)
    press_node = tank_nodes["press_tank"]

    figure, axis = plt.subplots(figsize=(11, 7))
    for node_id, node in history[0]["plant"].fluids.node.items():
        if "P" in node and node_id != press_node:
            axis.plot(time, [state["plant"].fluids.node[node_id]["P"] / 1e6 for state in history],
                      label=node_id.replace("_", " "))
    axis.set(xlabel="Time [s]", ylabel="Pressure [MPa]", title="System pressure ladder (COPV excluded)")
    _finish(figure, axis, paths["pressure_ladder"], burnout)

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(time, [state["plant"].fluids.node[press_node]["P"] / 1e6 for state in history],
              label="COPV pressure")
    axis.set(xlabel="Time [s]", ylabel="Pressure [MPa]", title="COPV blowdown")
    _finish(figure, axis, paths["copv_blowdown"], burnout)

    figure, axis = plt.subplots(figsize=(11, 6))
    for tank_id, node_id in tank_nodes.items():
        for key, phase in (("T", ""), ("T_liq", " liquid"), ("T_ull", " ullage")):
            if key in history[0]["plant"].fluids.node[node_id]:
                axis.plot(time, [state["plant"].fluids.node[node_id].get(key, np.nan) for state in history],
                          label=f"{tank_id.replace('_', ' ')}{phase}")
    axis.set(xlabel="Time [s]", ylabel="Temperature [K]", title="Tank temperatures")
    _finish(figure, axis, paths["tank_temperatures"], burnout)

    figure, axis = plt.subplots(figsize=(11, 6))
    for tank_id, label in (("ox_tank", "Oxidizer"), ("fuel_tank", "Fuel")):
        node_id = tank_nodes[tank_id]
        axis.plot(time, [state["plant"].fluids.node[node_id].get("m_liq", 0.0) for state in history],
                  label=label)
    axis.set(xlabel="Time [s]", ylabel="Liquid mass [kg]", title="Remaining propellant mass")
    _finish(figure, axis, paths["propellant_mass"], burnout)

    figure, pressure_axis = plt.subplots(figsize=(11, 6))
    thrust_axis = pressure_axis.twinx()
    pressure_axis.plot(time, _series(rows, "chamber_pressure_Pa", 1e-6), label="Chamber pressure")
    thrust_axis.plot(time, _series(rows, "thrust_N", 1 / 1000), color="tab:orange", label="Thrust")
    pressure_axis.set(xlabel="Time [s]", ylabel="Chamber pressure [MPa]", title="Chamber pressure and thrust")
    thrust_axis.set_ylabel("Thrust [kN]")
    _mark_burnout(pressure_axis, burnout)
    _finish(figure, [pressure_axis, thrust_axis], paths["pc_thrust"])

    figure, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)

    axes[0, 0].plot(time, _series(rows, "altitude_m"))
    axes[0, 0].set_ylabel("Altitude [m]")

    axes[0, 1].plot(time, _series(rows, "x_m"))
    axes[0, 1].set_ylabel("Downrange distance [m]")

    axes[1, 0].plot(time, _series(rows, "speed_m_s"), label="Speed")
    axes[1, 0].plot(time, _series(rows, "vx_m_s"), label="Horizontal velocity")
    axes[1, 0].plot(time, _series(rows, "vz_m_s"), label="Vertical velocity")
    axes[1, 0].set_ylabel("Velocity [m/s]")

    axes[1, 1].plot(time, _series(rows, "acceleration_m_s2"))
    axes[1, 1].set_ylabel("Vertical acceleration [m/s²]")

    axes[2, 0].plot(time, _series(rows, "mach"))
    axes[2, 0].set_ylabel("Mach")

    q = _series(rows, "dynamic_pressure_Pa", 1e-3)
    axes[2, 1].plot(time, q)
    max_q = int(np.nanargmax(q))
    axes[2, 1].plot(
        time[max_q],
        q[max_q],
        "o",
        label=f"Max Q: {q[max_q]:.1f} kPa at {time[max_q]:.1f} s",
    )
    axes[2, 1].set_ylabel("Dynamic pressure [kPa]")

    axes[3, 0].plot(time, _series(rows, "pitch_angle_deg"), label="Pitch angle")
    axes[3, 0].plot(
        time,
        _series(rows, "flight_path_angle_deg"),
        label="Flight-path angle",
    )
    axes[3, 0].plot(
        time,
        _series(rows, "angle_of_attack_deg"),
        label="Angle of attack",
    )
    axes[3, 0].set_ylabel("Angle [deg]")

    axes[3, 1].plot(time, _series(rows, "pitch_rate_rad_s"))
    axes[3, 1].set_ylabel("Pitch rate [rad/s]")

    for axis in axes[-1]:
        axis.set_xlabel("Time [s]")

    figure.suptitle("Kinematics")
    _finish(figure, axes, paths["kinematics"], burnout)

    figure, axis = plt.subplots(figsize=(8, 8))
    axis.plot(_series(rows, "x_m"), _series(rows, "altitude_m"))
    axis.set(
        xlabel="Downrange distance [m]",
        ylabel="Altitude [m]",
        title="Flight trajectory",
    )
    _finish(figure, axis, paths["trajectory"])

    thermal = history[0]["plant"].thermal
    if thermal is None:
        figure, axis = plt.subplots(figsize=(11, 7))
        axis.text(0.5, 0.5, "Axial wall temperatures unavailable\nExternal heating is disabled",
                  ha="center", va="center", transform=axis.transAxes)
        axis.set(title="Axial temperature profiles every 10 s")
        thermal_axes = axis
    else:
        figure, thermal_axes = plt.subplots(2, 1, figsize=(11, 10))
        map_axis, axis = thermal_axes
        node_ids = list(thermal.node)
        station = np.concatenate([thermal.node[node]["cells"]["station"] for node in node_ids])
        order = np.argsort(station)
        wall_history = np.asarray([
            np.concatenate([
                state["plant"].thermal.node[node]["cells"]["wall_T"] for node in node_ids
            ])[order]
            for state in history
        ])
        field = map_axis.pcolormesh(time, station[order], wall_history.T, shading="nearest")
        map_axis.set(xlabel="Time [s]", ylabel="Axial station [m]", title="Axial wall-temperature map")
        figure.colorbar(field, ax=map_axis, label="Wall temperature [K]")
        _mark_burnout(map_axis, burnout)
        targets = np.arange(10.0, time[-1] + 0.001, 10.0)
        indices = sorted({int(np.abs(time - target).argmin()) for target in targets})
        for index in indices:
            axis.plot(station[order], wall_history[index], label=f"t = {time[index]:g} s")
        if burnout is not None:
            index = int(np.abs(time - burnout).argmin())
            axis.plot(station[order], wall_history[index], "k--", linewidth=2,
                      label=f"Burnout: {time[index]:g} s")
        axis.set(xlabel="Axial station [m]", ylabel="Wall temperature [K]",
                 title="Axial profiles every 10 s")
    _finish(figure, thermal_axes, paths["axial_temperatures"])

    valve_ids = [branch_id for branch_id, branch in history[0]["plant"].fluids.branch.items()
                 if "is_open" in branch or "opening_fraction" in branch]
    plot_valve_actuations(history, paths["bang_bang"], burnout)

    regulators = [key for key in valve_ids if "opening_fraction" in history[0]["plant"].fluids.branch[key]]
    if regulators:
        paths["regulators"] = _path(base_path, "regulators")
        figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        for branch_id in regulators:
            for axis, field in zip(axes, ("opening_fraction", "pressure_error")):
                axis.plot(time, [state["plant"].fluids.branch[branch_id][field] for state in history],
                          label=branch_id.replace("_", " "))
        axes[0].set(ylabel="Opening fraction", title="Ideal pressure regulators")
        axes[1].set(xlabel="Time [s]", ylabel="Tank pressure − target [Pa]")
        _finish(figure, axes, paths["regulators"], burnout)

    figure, axis = plt.subplots(figsize=(11, 6))
    for branch_id in valve_ids:
        axis.plot(time, [state["plant"].fluids.mdot[branch_id] for state in history],
                  label=branch_id.replace("_BANGBANG", "").replace("_", " "))
    axis.set(xlabel="Time [s]", ylabel="Pressurant mass flow [kg/s]", title="Pressurant mass flow")
    _finish(figure, axis, paths["pressurant_mdot"], burnout)

    figure, flow_axis = plt.subplots(figsize=(11, 6))
    mr_axis = flow_axis.twinx()
    for name, label in (("oxidizer_mdot_kg_s", "Oxidizer"), ("fuel_mdot_kg_s", "Fuel"),
                        ("nozzle_mdot_kg_s", "Nozzle")):
        flow_axis.plot(time, _series(rows, name), label=label)
    mr_axis.plot(time, _series(rows, "mixture_ratio"), color="black", linestyle=":", label="Mixture ratio")
    flow_axis.set(xlabel="Time [s]", ylabel="Mass flow [kg/s]", title="Engine mass flow and mixture ratio")
    mr_axis.set_ylabel("Mixture ratio")
    _mark_burnout(flow_axis, burnout)
    _finish(figure, [flow_axis, mr_axis], paths["engine_mdot_mr"])

    station = np.asarray(history[0]["mass_properties"]["station"], dtype=float)
    mass = np.asarray([state["mass_properties"]["axial_mass"] for state in history])
    figure, axis = plt.subplots(figsize=(11, 7))
    field = axis.pcolormesh(time, station, mass.T, shading="nearest")
    figure.colorbar(field, ax=axis, label="Mass per axial cell [kg]")
    axis.plot(time, _series(rows, "cg_m"), color="white", linewidth=2, label="Center of mass")
    axis.plot(time, _series(rows, "cp_m"), color="black", linewidth=2, label="Center of pressure")
    axis.set(xlabel="Time [s]", ylabel="Axial station [m]", title="Vehicle mass distribution")
    _finish(figure, axis, paths["mass_distribution"], burnout)

    axial = np.asarray([state["loads"]["axial"] for state in history]) / 1000.0
    figure = plt.figure(figsize=(12, 8))
    axis = figure.add_subplot(111, projection="3d")
    surface = _load_surface(
        axis,
        time,
        station,
        axial,
        "Axial force distribution",
        "Internal axial force [kN]",
        burnout,
    )
    figure.colorbar(surface, ax=axis, shrink=0.65, pad=0.12, label="Internal axial force [kN]")
    _finish(figure, axis, paths["axial_force_distribution"])

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(13, 16),
        subplot_kw={"projection": "3d"},
    )
    for axis, key, scale, title, label in (
        (axes[0], "normal", 1 / 1000.0, "Normal-force distribution",
         "Distributed normal force [kN/cell]"),
        (axes[1], "shear", 1 / 1000.0, "Internal shear distribution",
         "Internal shear [kN]"),
        (axes[2], "bending", 1 / 1000.0, "Internal bending-moment distribution",
         "Internal bending moment [kN m]"),
    ):
        values = np.asarray([state["loads"][key] for state in history]) * scale
        surface = _load_surface(axis, time, station, values, title, label, burnout)
        figure.colorbar(surface, ax=axis, shrink=0.65, pad=0.12, label=label)
    _finish(figure, axes, paths["normal_force_distribution"])
    return paths
