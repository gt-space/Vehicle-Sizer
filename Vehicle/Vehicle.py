import numpy as np
from Fluids.design import initial_conditions, tank_design_pressure

from .COPV import COPV
from .Engine import Engine
from .Material import MaterialProperties
from .sections.AviBay import AviBay
from .sections.FinCan import FinCan
from .sections.InterTank import InterTank
from .sections.Nosecone import Nosecone
from .sections.PressTank import PressTank
from .sections.PropTank import PropTank
from .geometry_constraints import geometry_constraints, GeometryError

class Vehicle:

    def __init__(self, cfg: dict, fluid_properties=None):

        self.cfg: dict = cfg
        self.fluid_properties = fluid_properties
        self.engine = None
        self.dx: float = float(cfg["vehicle"]["dx"])
        self.initial_conditions = initial_conditions(cfg)
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

    def build(self, engine: Engine):
        """Build the configured nose-to-aft section stack."""

        self.engine = engine
        self.sections = self._build_sections()
        self.geometry_constraints = geometry_constraints(self)
        # A longer engine would extend outside the modeled OML and aero deck.
        # Return its signed constraint instead of silently truncating its mass.
        if any(value is not None and value < 0 for value in self.geometry_constraints.values()):
            raise GeometryError(self.geometry_constraints)
        self._stack_sections()
        fins = [s for s in self.sections if isinstance(s, FinCan)]
        if len(fins) != 1:
            raise ValueError("Vehicle requires exactly one fin can to locate the engine")
        self.engine_start_station = fins[0].start_station
        self.engine_end_station = self.engine_start_station + engine.length
        self.cell_edges = np.concatenate([s.cell_edges[:-1] for s in self.sections] + [np.array([self.length])])
        self.cell_widths = np.diff(self.cell_edges)
        self.engine_mass = engine.mass_on_grid(self.cell_edges, self.engine_start_station)
        self.dry_mass = np.concatenate([s.mass for s in self.sections]) + self.engine_mass
        self._assemble_vectors()
        self.get_mass_properties()
        return self.geometry_constraints

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
                    max_pressure=self._max_tank_pressure(tank_id),
                    t_wall_min=float(self.cfg["advanced"]["t_wall_min"]),
                    passthrough_diameter=definition["passthrough_diameter"],
                    passthrough_wall_thickness=definition[
                        "passthrough_wall_thickness"
                    ],
                    ellipse_ratio=definition["ellipse_ratio"],
                    ullage_factor=definition["ullage_factor"],
                    tank_id=tank_id,
                    weld_efficiency=definition.get("weld_efficiency", self.cfg["advanced"].get("weld_efficiency", 1.0)),
                )
            elif tank_type == "pressurant":
                if "volume" in definition and "volume_liters" in definition:
                    raise ValueError(f"Tank {tank_id!r}: specify volume (m^3) or volume_liters, not both")
                material = MaterialProperties.from_name(definition["material"])
                copv = COPV(
                    volume=(float(definition["volume"]) if "volume" in definition
                            else float(definition["volume_liters"]) * 1.0e-3),
                    diameter=float(definition["outer_diameter"]),
                    wall_thickness=float(definition["wall_thickness"]),
                    ellipse_ratio=float(definition["ellipse_ratio"]),
                    material_density=material.density,
                    mass=float(definition["mass"]) if "mass" in definition else None,
                )
                tanks[tank_id] = PressTank(self.cfg, copv, tank_id=tank_id)
            else:
                raise ValueError(
                    f"Unknown tank type {tank_type!r} for tank {tank_id!r}"
                )
        return tanks

    def _max_tank_pressure(self, tank_id: str) -> float:
        """Return the absolute pressure used to size a propellant tank wall."""

        initial_pressure = float(self.initial_conditions[tank_id]["P"])
        design_pressure = tank_design_pressure(self.cfg, tank_id, fallback=initial_pressure)
        nominal = max(design_pressure, initial_pressure)
        prop_system = self.cfg["prop_system"]
        legacy = prop_system.get("press_model")
        pressurization = prop_system.get(
            "pressurization",
            "blowdown" if legacy == "blowdown" else "bang_bang",
        )
        controls = [
            definition
            for definition in prop_system.get("bang_bang", {}).values()
            if definition.get("tank_id") == tank_id
        ] if pressurization == "bang_bang" else []
        template = prop_system.get("template")
        if template is not None:
            target_nodes = {key for key, node in template["nodes"].items() if node.get("tank_id") == tank_id}
            controls = [branch for branch in template["branches"].values()
                        if branch.get("component") == "bang_bang_valve" and branch["to"] in target_nodes]
        return nominal + max(float(control["pressure_band"]) for control in controls) if controls else 1.1 * nominal

    def _build_sections(self) -> list:
        sections = []
        used_tanks = set()
        for definition in self.cfg["vehicle"]["sections"]:
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
                    area_moment_of_inertia=definition["area_moment_of_inertia"],
                    feed_system_mass=definition.get("feed_system_mass", 0.0),
                    avi_mass=definition.get("avi_mass", 0.0),
                    mass=definition.get("mass"),
                )
            elif section_type == "fin_can":
                section = FinCan(self.cfg, self.engine)
            else:
                raise ValueError(f"Unknown vehicle section type {section_type!r}")
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

    def _assemble_vectors(self):
        self.station = np.concatenate([sec.station for sec in self.sections])
        self.mass = np.concatenate([sec.mass for sec in self.sections])
        if hasattr(self, "engine_mass"):
            self.mass = self.mass + self.engine_mass
        self.EI = np.concatenate([sec.EI for sec in self.sections])
        self.lat_area = np.concatenate([sec.lat_area for sec in self.sections])
        self.surf_area = np.concatenate([sec.surf_area for sec in self.sections])

    def get_mass_properties(self):
        self.total_mass = np.sum(self.mass)
        self.cg = np.sum(self.mass * self.station) / self.total_mass
        self.Ixx = sum(sec.Ixx for sec in self.sections)
        self.Iyy = sum(
            sec.Iyy + np.sum(sec.mass) * (sec.cg - self.cg)**2
            for sec in self.sections
        )
        if hasattr(self, "engine_mass"):
            self.Iyy += np.sum(self.engine_mass * (self.station - self.cg)**2)
            # Retain the existing shell-radius roll-inertia approximation.
            self.Ixx += np.sum(self.engine_mass) * (self.cfg["vehicle"]["OMLD"] / 2)**2

    def update_mass_distribution(self, node_states: dict) -> None:
        """Apply fluid-network axial mass vectors and refresh mass properties."""

        tank_sections = {
            section.tank_id: section
            for section in self.sections
            if hasattr(section, "tank_id")
        }
        if len(tank_sections) != sum(
            hasattr(section, "tank_id") for section in self.sections
        ):
            raise ValueError("Vehicle tank IDs must be unique")

        fluid_tanks = {}
        for node_id, state in node_states.items():
            if "axial_mass" not in state:
                continue
            tank_id = state.get("tank_id")
            if tank_id is None:
                raise ValueError(
                    f"Fluid node '{node_id}' with axial mass requires a tank_id"
                )
            if tank_id in fluid_tanks:
                raise ValueError(f"Multiple fluid nodes reference tank '{tank_id}'")
            fluid_tanks[tank_id] = state["axial_mass"]

        missing_sections = set(fluid_tanks) - set(tank_sections)
        if missing_sections:
            raise ValueError(
                f"Fluid nodes reference unknown vehicle tanks: {sorted(missing_sections)}"
            )
        missing_states = set(tank_sections) - set(fluid_tanks)
        if missing_states:
            raise ValueError(
                f"Vehicle tanks have no fluid-node state: {sorted(missing_states)}"
            )

        for tank_id, axial_mass in fluid_tanks.items():
            tank_sections[tank_id].set_fluid_mass(axial_mass)
        self._assemble_vectors()
        self.get_mass_properties()

    def aero_candidate(self) -> dict:
        """Return the assembled geometry expected by the RASAero model."""

        if self.engine is None or self.length is None:
            raise RuntimeError("Vehicle must be built before exporting aero geometry")
        fin = next((section for section in self.sections if isinstance(section, FinCan)), None)
        if fin is None:
            raise ValueError("Aerodynamic model requires a fin-can section")
        metres_per_inch = 0.0254
        return {
            "omld": float(self.cfg["vehicle"]["OMLD"]) / metres_per_inch,
            "length": float(self.length) / metres_per_inch,
            "fineness": float(self.cfg["nosecone"]["fineness_ratio"]),
            "exit": 2.0 * np.sqrt(self.engine.exit_area / np.pi) / metres_per_inch,
            "boattail_aft": fin.boattail_aft_diameter / metres_per_inch,
            "boattail_length": fin.length / metres_per_inch,
            "span": fin.span / metres_per_inch,
            "root": fin.root_chord / metres_per_inch,
            "tip": fin.tip_chord / metres_per_inch,
            "sweep_fraction": fin.sweep_fraction,
            "thickness": fin.fin_thickness / metres_per_inch,
        }

# Legacy analytical aero / unused input container (inactive).
#     def get_CNa(self, M: float, alpha: float):
#         for sec in self.sections:
#             sec.get_CNa(M, alpha)
#         self.CNa = np.concatenate([sec.CNa for sec in self.sections])
#         self.cp = np.sum(self.CNa * self.station) / np.sum(self.CNa)

    # def update(self):
