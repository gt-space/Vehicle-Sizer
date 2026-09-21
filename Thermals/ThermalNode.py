from __future__ import annotations

from typing import Any, Dict

import numpy as np

from simulation_types import AeroOut, AtmosState, FluidOut, KinematicsState
from . import heating
from .ThermalCircuit import solve_lumped_wall

SIGMA = 5.670374419e-8


class DryNodeModel:
    """Wall exposed externally with no fluid-side heat-transfer path."""

    def fluid_boundary(
        self,
        node,
        fluid_out: FluidOut,
        wall_T: np.ndarray,
        axial_specific_force: float,
    ):
        del fluid_out, wall_T, axial_specific_force
        zeros = np.zeros(node.n)
        return (
            zeros,
            zeros,
            zeros,
            zeros,
            np.full(node.n, "dry", dtype=object),
            {},
            None,
        )


def natural_convection_htc(
    delta_T: np.ndarray,
    acceleration: float,
    length: np.ndarray,
    rho: np.ndarray,
    mu: np.ndarray,
    conductivity: np.ndarray,
    specific_heat: np.ndarray,
    expansion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Churchill-Chu natural-convection HTC for an axial wall surface."""

    if np.any(np.asarray([rho, mu, conductivity, specific_heat, expansion]) <= 0.0):
        raise ValueError("Natural-convection fluid properties must be positive")
    kinematic_viscosity = mu / rho
    grashof = (
        abs(float(acceleration))
        * expansion
        * np.abs(delta_T)
        * length**3
        / kinematic_viscosity**2
    )
    prandtl = specific_heat * mu / conductivity
    rayleigh = grashof * prandtl
    nusselt = (
        0.825
        + 0.387
        * rayleigh ** (1.0 / 6.0)
        / (1.0 + (0.492 / prandtl) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    ) ** 2
    return nusselt * conductivity / length, grashof


class WetNodeModel(DryNodeModel):
    """Wall coupled to one bulk fluid temperature per contacting phase."""

    def __init__(self, insulated: bool = False) -> None:
        self.insulated = bool(insulated)

    def _phases(self, node, fluid_out: FluidOut):
        matches = [
            (fluid_node_id, output)
            for fluid_node_id, output in fluid_out.node.items()
            if output.get("tank_id") == node.id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Thermal tank {node.id!r} requires exactly one fluid-node output"
            )
        fluid_node_id, fluid_node = matches[0]

        phase_states = fluid_node.get("phase_states")
        fluids = phase_states if phase_states else fluid_node.get("fluids", {})
        phases = {}
        for name, fluid in fluids.items():
            phase = str(name if phase_states else fluid["phase"])
            phases[phase] = fluid
        if not phases:
            raise ValueError(f"Fluid node {node.id!r} exposes no phase temperatures")

        labels = np.full(node.n, next(iter(phases)), dtype=object)
        if "liquid" in phases and "gas" in phases and "fill_height" in fluid_node:
            fill_height = float(fluid_node["fill_height"])
            interface = node.section.end_station - fill_height
            edge = node.cell_edges[np.argmin(np.abs(node.cell_edges - interface))]
            labels[:] = "gas"
            labels[node.cell_centers >= edge] = "liquid"
        elif "liquid" in phases and "gas" in phases:
            raise ValueError(
                f"Two-phase thermal node {node.id!r} requires fill_height"
            )
        return fluid_node_id, phases, labels

    def fluid_boundary(
        self,
        node,
        fluid_out: FluidOut,
        wall_T: np.ndarray,
        axial_specific_force: float,
    ):
        fluid_node_id, phases, labels = self._phases(node, fluid_out)
        fluid_T = np.asarray([phases[phase]["T"] for phase in labels], dtype=float)
        active_h = np.zeros(node.n)
        grashof = np.zeros(node.n)
        if not self.insulated:
            required = ("rho", "mu", "k", "cp", "beta")
            for phase, fluid in phases.items():
                missing = [name for name in required if name not in fluid]
                if missing:
                    raise ValueError(
                        f"Thermal node {node.id!r} phase {phase!r} is missing "
                        f"properties {missing}"
                    )
            widths = np.diff(node.cell_edges)
            phase_length = {
                phase: float(np.sum(widths[labels == phase])) for phase in phases
            }
            length = np.asarray([phase_length[phase] for phase in labels])
            properties = {
                name: np.asarray([phases[phase][name] for phase in labels], dtype=float)
                for name in required
            }
            active_h, grashof = natural_convection_htc(
                wall_T - fluid_T,
                axial_specific_force,
                length,
                properties["rho"],
                properties["mu"],
                properties["k"],
                properties["cp"],
                properties["beta"],
            )
        conductance = np.zeros(node.n)
        active = active_h > 0.0
        area = node.internal_area[active]
        conductance[active] = 1.0 / (
            node.thickness / (2.0 * node.conductivity * area)
            + 1.0 / (active_h[active] * area)
        )
        phase_data = {
            phase: {
                "temperature": float(fluid["T"]),
                "htc": float(np.mean(active_h[labels == phase]))
                if np.any(labels == phase)
                else 0.0,
                "grashof": float(np.max(grashof[labels == phase]))
                if np.any(labels == phase)
                else 0.0,
            }
            for phase, fluid in phases.items()
        }
        return (
            conductance,
            fluid_T,
            active_h,
            grashof,
            labels,
            phase_data,
            fluid_node_id,
        )


class ThermalNode:
    """Axial wall cells sharing one dry or wet boundary model."""

    def __init__(
        self,
        node_id: str,
        section: Any,
        model: DryNodeModel,
        initial_T: float,
        sink_T: float,
        density: float,
        specific_heat: float,
        conductivity: float,
    ) -> None:
        self.id = node_id
        self.section = section
        self.model = model
        self.n = int(section.n)
        self.wall_T = np.full(self.n, float(initial_T))
        self.sink_T = float(sink_T)
        self.density = float(density)
        self.specific_heat = float(specific_heat)
        self.conductivity = float(conductivity)
        self.thickness = float(section.wall_thickness)
        self.emissivity = float(section.emissivity)
        self.oml_area = np.asarray(section.get_thermal_oml_area(), dtype=float)
        self.shell_mass = np.asarray(section.get_thermal_shell_mass(), dtype=float)
        self.internal_area = np.asarray(section.get_thermal_internal_area(), dtype=float)
        self.cell_edges = getattr(section, "cell_edges", np.linspace(section.start_station, section.end_station, self.n + 1)).copy()
        self.cell_centers = 0.5 * (self.cell_edges[:-1] + self.cell_edges[1:])
        for name, values in (
            ("OML area", self.oml_area),
            ("shell mass", self.shell_mass),
            ("internal area", self.internal_area),
        ):
            if values.shape != (self.n,) or np.any(values <= 0.0):
                raise ValueError(f"Thermal node {node_id!r} requires positive {name} per cell")
        if min(self.density, self.specific_heat, self.conductivity, self.thickness) <= 0.0:
            raise ValueError(f"Thermal properties for node {node_id!r} must be positive")

    def _external_heat_flux(
        self,
        kin: KinematicsState,
        atm: AtmosState,
        wall_T: np.ndarray,
    ) -> np.ndarray:
        if self.section.__class__.__name__ == "Nosecone":
            return heating.get_nose_heating(
                atm,
                float(self.section.cfg["nosecone"].get("tip_radius", 0.01)),
                self.section.radius,
                self.section.dx,
                wall_T,
            )
        x = np.maximum(self.section.station, 0.5 * self.section.dx)
        return heating.get_body_heating(x, wall_T, atm, abs(kin.alpha))

    def trial(
        self,
        kin: KinematicsState,
        atm: AtmosState,
        aero_out: AeroOut,
        fluid_out: FluidOut,
        wall_T_guess: np.ndarray | None = None,
        axial_specific_force: float = 0.0,
    ) -> Dict[str, Any]:
        del aero_out  # Reserved for future local aerodynamic boundary data.
        guess = self.wall_T if wall_T_guess is None else np.asarray(wall_T_guess, dtype=float)
        if guess.shape != (self.n,):
            raise ValueError(f"Thermal node {self.id!r} received the wrong wall-temperature shape")

        heat_flux = self._external_heat_flux(kin, atm, guess)
        heat_aero = heat_flux * self.oml_area
        (
            fluid_G,
            fluid_T,
            internal_h,
            grashof,
            phase,
            phase_data,
            fluid_node_id,
        ) = self.model.fluid_boundary(
            self, fluid_out, guess, axial_specific_force
        )

        radiation_G = (
            self.emissivity
            * SIGMA
            * self.oml_area
            * (guess + self.sink_T)
            * (guess**2 + self.sink_T**2)
        )
        total_G = radiation_G + fluid_G
        boundary_T = np.divide(
            radiation_G * self.sink_T + fluid_G * fluid_T,
            total_G,
            out=np.zeros(self.n),
            where=total_G > 0.0,
        )
        wall_T = solve_lumped_wall(
            kin.dt,
            self.wall_T,
            self.shell_mass * self.specific_heat,
            heat_aero,
            total_G,
            boundary_T,
        )
        heat_radiation = self.emissivity * SIGMA * self.oml_area * (
            wall_T**4 - self.sink_T**4
        )
        heat_to_fluid = fluid_G * (wall_T - fluid_T)

        recovery_T = heating.get_recovery_temperature(atm, abs(kin.alpha))
        external_h = np.abs(heat_flux) / np.maximum(np.abs(recovery_T - guess), 1.0)
        exposed_area = self.oml_area + np.where(internal_h > 0.0, self.internal_area, 0.0)
        characteristic_length = (self.shell_mass / self.density) / exposed_area
        biot = np.maximum(external_h, internal_h) * characteristic_length / self.conductivity

        for name, output in phase_data.items():
            output["heat_rate"] = float(np.sum(heat_to_fluid[phase == name]))

        output = {
            "cells": {
                "station": self.section.station.copy(),
                "wall_T": wall_T,
                "phase": phase,
                "biot": biot,
                "internal_htc": internal_h,
                "grashof": grashof,
                "heat_aero": heat_aero,
                "heat_radiation": heat_radiation,
                "heat_to_fluid": heat_to_fluid,
            },
            "phases": phase_data,
        }
        if fluid_node_id is not None:
            output["fluid_node_id"] = fluid_node_id
        return output

    def commit(self, output: Dict[str, Any]) -> None:
        self.wall_T = np.asarray(output["cells"]["wall_T"], dtype=float).copy()
