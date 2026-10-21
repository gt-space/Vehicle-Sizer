from copy import deepcopy
from math import isfinite
import numpy as np

from .Engine import Engine
from .Material import MaterialProperties
from .sections.AviBay import AviBay
from .sections.FinCan import FinCan
from .sections.InterTank import InterTank
from .sections.Nosecone import Nosecone
from .sections.PressTank import PressTank
from .sections.PropTank import PropTank

class Vehicle:

    def __init__(self, cfg: dict, fluid_properties=None, *, initial_states, wall_pressures):

        self.cfg: dict = cfg
        self.fluid_properties = fluid_properties
        self.engine = None
        self.propulsion = None
        self.dx: float = float(cfg["vehicle"]["dx"])
        self.initial_conditions = initial_states
        self.wall_pressures = wall_pressures
        self.tanks = self._build_tanks()
        self.sections: list = []
        self.n: int = None

        self.station: np.ndarray = None
        self.mass: np.ndarray = None
        self.EI: np.ndarray = None
        self.lat_area: np.ndarray = None
        self.surf_area: np.ndarray = None
        self.CNa: np.ndarray = None

        self.length: float = None
        self.total_mass: float = None
        self.cg: float = None
        self.cp: float = None
        self.Ixx: float = None
        self.Iyy: float = None

    def build(self, propulsion):
        """Assemble the configured vehicle using the sized propulsion system."""

        self.propulsion = propulsion
        engine_cfg = self.cfg["engine"]
        self.engine = Engine(float(engine_cfg["mass"]), float(engine_cfg["length"]),
                             self.propulsion.exit_area)
        self.sections = self._build_sections()
        self._stack_sections()
        fins = [s for s in self.sections if isinstance(s, FinCan)]
        if len(fins) != 1:
            raise ValueError("Vehicle requires exactly one fin can to locate the engine")
        self.engine_start_station = fins[0].start_station
        self.engine_end_station = self.engine_start_station + self.engine.length
        self.cell_edges = np.concatenate([s.cell_edges[:-1] for s in self.sections] + [np.array([self.length])])
        self.cell_widths = np.diff(self.cell_edges)
        self.engine_mass = self.engine.mass_on_grid(self.cell_edges, self.engine_start_station)
        self.dry_mass = np.concatenate([s.mass for s in self.sections]) + self.engine_mass
        self.station = np.concatenate([sec.station for sec in self.sections])
        self.mass = self.dry_mass.copy()
        self.EI = np.concatenate([sec.EI for sec in self.sections])
        self.lat_area = np.concatenate([sec.lat_area for sec in self.sections])
        self.surf_area = np.concatenate([sec.surf_area for sec in self.sections])
        offsets = np.cumsum([0] + [sec.n for sec in self.sections])
        slices = {id(sec): slice(start, stop) for sec, start, stop in
                  zip(self.sections, offsets[:-1], offsets[1:])}
        self.fluid_sections = tuple((key, self.tanks[node['tank_id']], slices[id(self.tanks[node['tank_id']])])
                                    for key, node in propulsion.node_definitions.items() if 'tank_id' in node)
        if (len(self.fluid_sections) != len(self.tanks)
                or {id(sec) for _, sec, _ in self.fluid_sections} != {id(sec) for sec in self.tanks.values()}):
            raise ValueError('Every vehicle tank requires exactly one fluid node')
        self.dry_pitch_correction = 0.0
        if 'Iyy' in self.cfg['vehicle']:
            supplied = float(self.cfg['vehicle']['Iyy'])
            if not np.isfinite(supplied) or supplied <= 0:
                raise ValueError('vehicle.Iyy must be finite and positive (dry pitch inertia, kg m^2)')
            dry_cg = np.sum(self.dry_mass * self.station) / np.sum(self.dry_mass)
            self.dry_pitch_correction = supplied - np.sum(self.dry_mass * (self.station - dry_cg)**2)
        self.get_mass_properties()

    def _build_tanks(self) -> dict:
        """Create each tank section once for structural and fluid use."""

        tanks = {}
        state0 = self.initial_conditions
        for tank_id, definition in self.cfg["tanks"].items():
            tank_type = definition["type"]
            if tank_type == "propellant":
                if self.fluid_properties is None:
                    raise ValueError(
                        "Vehicle requires a fluid-property source to size propellant tanks"
                    )
                initial = state0[tank_id]
                liquid = self.fluid_properties.state_pt(
                    initial["fluid"],
                    float(initial["P"]),
                    float(initial["T"]),
                )
                tanks[tank_id] = PropTank(
                    self.cfg,
                    prop_mass=definition["propellant_mass"],
                    liquid_density=liquid.rho,
                    material=MaterialProperties.from_name(definition["material"]),
                    wall_thickness=definition.get("wall_thickness"),
                    max_pressure=self.wall_pressures[tank_id],
                    t_wall_min=float(self.cfg["advanced"]["t_wall_min"]),
                    passthrough_diameter=definition["passthrough_diameter"],
                    passthrough_wall_thickness=definition[
                        "passthrough_wall_thickness"
                    ],
                    ellipse_ratio=definition["ellipse_ratio"],
                    ullage_factor=definition.get("ullage_factor"),
                    volume=definition.get("volume"),
                    length=definition.get("length"),
                    tank_id=tank_id,
                    weld_efficiency=definition.get("weld_efficiency", self.cfg["advanced"].get("weld_efficiency", 1.0)),
                )
            elif tank_type == "pressurant":
                tanks[tank_id] = PressTank(self.cfg, tank_id=tank_id)
            else:
                raise ValueError(
                    f"Unknown tank type {tank_type!r} for tank {tank_id!r}"
                )
        return tanks

    def _resolve_section_masses(self):
        """Resolve calculated component masses without mutating the config."""
        definitions = deepcopy(self.cfg["vehicle"]["sections"])
        placements = [(i, d) for i, d in enumerate(definitions)
                      if "battery" in d.get("masses", {})]
        self.battery_sizing = {}
        if "battery" not in self.cfg["prop_system"]:
            if any(d["masses"]["battery"] == "auto" for _, d in placements):
                raise ValueError("battery: auto requires prop_system.battery sizing")
            return definitions
        if len(placements) != 1:
            raise ValueError("Battery sizing requires exactly one section with masses.battery")
        if self.propulsion is None:
            raise ValueError("Build vehicle with a sized PropSystem before resolving battery mass")
        battery_sizing = self.propulsion.battery_sizing
        if not battery_sizing:
            raise ValueError("PropSystem has no battery sizing for the configured battery")
        index, definition = placements[0]
        configured = definition["masses"]["battery"]
        automatic = configured == "auto"
        selected = battery_sizing["calculated_mass_kg"] if automatic else configured
        if isinstance(selected, bool):
            raise ValueError("Battery mass must be numeric or auto")
        selected = float(selected)
        if not isfinite(selected) or selected < 0:
            raise ValueError("Battery mass must be finite and non-negative")
        definition["masses"]["battery"] = selected
        override = definition.get("mass_override", definition.get("mass"))
        self.battery_sizing = dict(deepcopy(battery_sizing), section_index=index,
            section_type=definition["type"],
            mass_source="section_override" if override is not None else
                        ("calculated" if automatic else "battery_override"),
            selected_mass_kg=selected,
            used_mass_kg=selected if override is None else None,
            section_mass_override_kg=override)
        return definitions

    def _build_sections(self) -> list:
        sections = []
        used_tanks = set()
        names = set()
        for definition in self._resolve_section_masses():
            name = definition.get("name")
            if name is not None:
                if not isinstance(name, str) or not name.strip() or name in names:
                    raise ValueError("Vehicle section names must be nonempty and unique")
                names.add(name)
            section_type = definition["type"]
            if section_type == "nosecone":
                section = Nosecone(self.cfg)
            elif section_type == "avi_bay":
                section = AviBay(self.cfg)
            elif section_type in ("press_tank", "prop_tank"):
                tank_id = definition["tank_id"]
                if tank_id in used_tanks:
                    raise ValueError(f"Tank {tank_id!r} appears more than once")
                try:
                    section = self.tanks[tank_id]
                except KeyError as exc:
                    raise ValueError(f"Unknown vehicle tank {tank_id!r}") from exc
                expected = PressTank if section_type == "press_tank" else PropTank
                if not isinstance(section, expected):
                    raise ValueError(
                        f"Section type {section_type!r} does not match tank "
                        f"{tank_id!r}"
                    )
                used_tanks.add(tank_id)
            elif section_type == "inter_tank":
                section = InterTank(
                    self.cfg,
                    length=definition["length"],
                    stiffness_EI=definition["stiffness_EI"],
                )
            elif section_type == "fin_can":
                section = FinCan(self.cfg, self.engine)
            else:
                raise ValueError(f"Unknown vehicle section type {section_type!r}")
            section.stiffness_input = definition.get("stiffness_EI")
            section.name = name
            section.mass_inputs = definition
            sections.append(section)

        missing = set(self.tanks) - used_tanks
        if missing:
            raise ValueError(f"Configured tanks missing from vehicle stack: {sorted(missing)}")
        return sections

    def _stack_sections(self):

        x_current = 0.0

        for sec in self.sections:

            sec.start_station = x_current
            sec.end_station = x_current + sec.length
            sec.cell_edges = sec.start_station + sec.local_edges
            sec.station = sec.start_station + sec.local_centers
            sec.build()

            x_current = sec.end_station

        self.length = x_current
        self.n = sum(sec.n for sec in self.sections)

    def get_mass_properties(self):
        self.total_mass = np.sum(self.mass)
        self.cg = np.sum(self.mass * self.station) / self.total_mass
        self.Ixx = sum(sec.Ixx for sec in self.sections)
        self.Iyy = sum(
            sec.Iyy + np.sum(sec.mass) * (sec.cg - self.cg)**2
            for sec in self.sections
        )
        self.Iyy += np.sum(self.engine_mass * (self.station - self.cg)**2)
        # Retain the existing shell-radius roll-inertia approximation.
        self.Ixx += np.sum(self.engine_mass) * (self.cfg["vehicle"]["OMLD"] / 2)**2
        self.Iyy += self.dry_pitch_correction

    def update_mass_distribution(self, node_states: dict) -> None:
        """Apply changing inventories through the fixed node/section/grid mapping."""
        for node_id, section, cells in self.fluid_sections:
            axial_mass = np.asarray(node_states[node_id]['axial_mass'], dtype=float)
            if axial_mass.shape != section.dry_mass.shape:
                raise ValueError(f"Tank '{section.tank_id}' requires {section.n} axial mass values")
            if np.any(~np.isfinite(axial_mass)) or np.any(axial_mass < 0):
                raise ValueError(f"Tank '{section.tank_id}' fluid mass must be finite and nonnegative")
            section.mass = section.dry_mass + axial_mass
            section.get_MOI()
            self.mass[cells] = section.mass + self.engine_mass[cells]
        self.get_mass_properties()

    def aero_candidate(self) -> dict:
        """Return the assembled geometry expected by the RASAero model."""

        if self.engine is None or self.length is None:
            raise RuntimeError("Vehicle must be built before exporting aero geometry")
        fin = next((section for section in self.sections if isinstance(section, FinCan)), None)
        if fin is None:
            raise ValueError("Aerodynamic model requires a fin-can section")
        metres_per_inch = 0.0254
        exit_diameter = self.cfg.get("engine", {}).get("exit_diameter")
        if exit_diameter is None:
            exit_diameter = 2.0 * np.sqrt(self.engine.exit_area / np.pi)
        exit_diameter = float(exit_diameter)
        if not np.isfinite(exit_diameter) or exit_diameter <= 0:
            raise ValueError("engine.exit_diameter must be finite and positive (m)")
        return {
            "omld": float(self.cfg["vehicle"]["OMLD"]) / metres_per_inch,
            "length": float(self.length) / metres_per_inch,
            "fineness": float(self.cfg["nosecone"]["fineness_ratio"]),
            "exit": exit_diameter / metres_per_inch,
            "boattail_aft": fin.boattail_aft_diameter / metres_per_inch,
            "boattail_length": fin.length / metres_per_inch,
            "span": fin.span / metres_per_inch,
            "root": fin.root_chord / metres_per_inch,
            "tip": fin.tip_chord / metres_per_inch,
            "sweep_fraction": fin.sweep_fraction,
            "thickness": fin.fin_thickness / metres_per_inch,
        }
