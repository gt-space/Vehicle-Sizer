from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np

from simulation_types import AeroOut, AtmosState, FluidOut, KinematicsState, ThermalOut
from Vehicle.Material import MaterialProperties
from .ThermalNode import DryNodeModel, ThermalNode, WetNodeModel
from Thermals.heat_sources import thermal_model


class ThermalNetwork:
    """Build and evaluate section-level arrays of independent axial wall cells."""

    def __init__(self, cfg: Dict[str, Any], vehicle: Any, selections=None) -> None:
        thermal_cfg = cfg.get("thermal", {})
        if 'external_heating' in thermal_cfg or 'nodes' in thermal_cfg:
            raise ValueError('Select thermal.model per dynamic model; legacy external_heating/nodes settings are unsupported')
        if selections is None:
            selections = {key: tank.get('thermal', {}) for key, tank in cfg.get('tanks', {}).items()}
        selected = {key: value for key, value in selections.items() if thermal_model(value) == 'Aeroheating'}
        missing = {"initial_temperature", "sink_temperature"}.difference(thermal_cfg)
        if missing:
            raise ValueError(f"Thermal network requires explicit temperatures: {sorted(missing)}")
        try:
            initial_T = float(thermal_cfg["initial_temperature"])
            sink_T = float(thermal_cfg["sink_temperature"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Thermal temperatures must be finite and positive") from exc
        if not np.isfinite([initial_T, sink_T]).all() or min(initial_T, sink_T) <= 0:
            raise ValueError("Thermal temperatures must be finite and positive")
        overrides = thermal_cfg.get("material_overrides", {})
        counts = defaultdict(int)
        self.nodes: Dict[str, ThermalNode] = {}

        for section in vehicle.sections:
            node_id = getattr(section, "tank_id", None)
            if node_id is None:
                kind = section.__class__.__name__.lower()
                node_id = f"{kind}_{counts[kind]}"
                counts[kind] += 1
            if node_id not in selected:
                continue
            material = MaterialProperties.from_name(section.wall_material)
            material_cfg = overrides.get(section.wall_material, {})

            def thermal_property(name: str) -> float:
                value = material_cfg.get(name, getattr(material, name))
                if value is None:
                    raise ValueError(
                        f"Thermal material {section.wall_material!r} requires {name}"
                    )
                return float(value)

            if hasattr(section, "tank_id"):
                model = WetNodeModel()
            else:
                model = DryNodeModel()
            wall_temperature = float(selected[node_id].get('initial_wall_temperature', initial_T))
            if not np.isfinite(wall_temperature) or wall_temperature <= 0:
                raise ValueError(f'Invalid initial wall temperature for {node_id!r}')
            self.nodes[node_id] = ThermalNode(
                node_id=node_id,
                section=section,
                model=model,
                initial_T=wall_temperature,
                sink_T=sink_T,
                density=material.density,
                specific_heat=thermal_property("specific_heat"),
                conductivity=thermal_property("thermal_conductivity"),
            )
        if selected.keys() - self.nodes.keys():
            raise ValueError(f'Aeroheating targets have no wall geometry: {sorted(selected.keys() - self.nodes.keys())}')

    def trial(
        self,
        kin: KinematicsState,
        atm: AtmosState,
        aero_out: AeroOut,
        fluid_out: FluidOut,
        previous: Optional[ThermalOut] = None,
        axial_specific_force: float = 0.0,
    ) -> ThermalOut:
        output = {}
        for node_id, node in self.nodes.items():
            guess = None
            if previous is not None:
                guess = previous.node[node_id]["cells"]["wall_T"]
            output[node_id] = node.trial(
                kin,
                atm,
                aero_out,
                fluid_out,
                guess,
                axial_specific_force=axial_specific_force,
            )
        return ThermalOut(node=output)

    def commit(self, thermal_out: ThermalOut) -> None:
        if thermal_out.node.keys() != self.nodes.keys():
            raise ValueError("Thermal output does not match the configured network")
        for node_id, node in self.nodes.items():
            node.commit(thermal_out.node[node_id])

    @property
    def wall_T(self) -> np.ndarray:
        return np.concatenate([node.wall_T for node in self.nodes.values()]) if self.nodes else np.array([])
