import numpy as np
from .Engine import Engine

class Vehicle:

    def __init__(self, cfg: dict, engine: Engine, sections: list):

        self.cfg: dict = cfg
        self.engine = engine
        self.sections: list = sections
        self.dx: float = cfg["vehicle"]["dx"]
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

    def build(self):
        self.engine.build()
        self._stack_sections()
        self._assemble_vectors()
        self.get_mass_properties()

    def _stack_sections(self):

        x_current = 0.0

        for sec in self.sections:

            sec.start_station = x_current
            sec.end_station = x_current + sec.length
            sec.station = sec.start_station + np.arange(sec.n) * sec.dx
            sec.build()

            x_current = sec.end_station

        self.length = x_current
        self.n = int(np.ceil(self.length / self.dx))

    def _assemble_vectors(self):
        self.station = np.concatenate([sec.station for sec in self.sections])
        self.mass = np.concatenate([sec.mass for sec in self.sections])
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

    def get_CNa(self, M: float, alpha: float):
        for sec in self.sections:
            sec.get_CNa(M, alpha)
        self.CNa = np.concatenate([sec.CNa for sec in self.sections])
        self.cp = np.sum(self.CNa * self.station) / np.sum(self.CNa)

    # def update(self):
