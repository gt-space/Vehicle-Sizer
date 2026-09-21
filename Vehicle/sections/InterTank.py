import numpy as np
from scipy.optimize import brentq
from .Section import Section
from ..Material import MaterialProperties
from ..utils import distribute as dist
# from ..utils import aero  # Legacy analytical aero disabled.
from ..utils import geometry as geo

class InterTank(Section):

    def __init__(self, cfg: dict, length: float, area_moment_of_inertia: float,
                 *, feed_system_mass: float = 0.0, avi_mass: float = 0.0,
                 mass: float | None = None):

        super().__init__(cfg)
        self.mass_override = None if mass is None else float(mass)
        if self.mass_override is not None and (
            not np.isfinite(self.mass_override) or self.mass_override <= 0.0
        ):
            raise ValueError("Intertank mass override must be finite and positive")
        self.feed_system_mass = float(feed_system_mass)
        self.avi_mass = float(avi_mass)
        if any(not np.isfinite(mass) or mass < 0.0
               for mass in (self.feed_system_mass, self.avi_mass)):
            raise ValueError("Intertank hardware masses must be finite and nonnegative")
        self.length = length
        self.set_grid()
        self.area_moment_of_inertia = float(area_moment_of_inertia)
        if self.area_moment_of_inertia <= 0.0:
            raise ValueError("Intertank area moment of inertia must be positive")
        self.stringer_count = int(cfg["inter_tank"]["stringer_count"])
        if self.stringer_count < 3:
            raise ValueError("Intertank requires at least three stringers")
        self.wall_thickness = cfg["inter_tank"]["clamshell_wall_thickness"]
        self.wall_material = cfg["inter_tank"]["clamshell_material"]
        self.emissivity = 0.85

    def get_mass(self):
        # Geometry still determines stiffness and the thermal shell distribution.
        self.stringer_thickness = self._get_stringer_thickness()
        self.shell_mass = dist.uniform(self._get_clamshell_mass(), self.n)
        mass = self.mass_override
        if mass is None:
            density = MaterialProperties.from_name(self.cfg["inter_tank"]["stringer_material"]).density
            stringer_mass = self.stringer_thickness**2 * self.length * self.stringer_count * density
            mass = np.sum(self.shell_mass) + stringer_mass + self.feed_system_mass + self.avi_mass
        self.mass = dist.uniform(mass, self.n)

    def _get_clamshell_mass(self) -> float:
        t = self.cfg["inter_tank"]["clamshell_wall_thickness"]
        r_o = self.cfg["vehicle"]["OMLD"] * 0.5
        r_i = r_o - t
        rho = MaterialProperties.from_name(
            self.cfg["inter_tank"]["clamshell_material"]
        ).density
        return geo.annulus_volume(r_o, r_i, self.length) * rho

    def _get_stringer_thickness(self) -> float:
        required = self.area_moment_of_inertia

        radius = (
            0.5 * self.cfg["vehicle"]["OMLD"]
            - self.cfg["inter_tank"]["clamshell_wall_thickness"]
        )

        def error(a: float) -> float:
            centroid_radius = radius - 0.5 * a
            inertia = self.stringer_count * (
                a**4 / 12.0 + 0.5 * a**2 * centroid_radius**2
            )
            return inertia - required

        upper = 2.0 * radius
        if error(upper) < 0.0:
            raise ValueError("Requested intertank inertia cannot fit in its diameter")
        return brentq(error, 0.0, upper)

    def get_EI(self):
        self.EI = dist.uniform_full(
            self._get_stringer_EI(self.stringer_thickness),
            self.n,
        )

    def _get_stringer_EI(self, a: float) -> float:
        t = self.cfg["inter_tank"]["clamshell_wall_thickness"]
        r_o = self.cfg["vehicle"]["OMLD"] * 0.5
        r = r_o - t - a * 0.5
        I = self.stringer_count * (a**4 / 12.0 + 0.5 * a**2 * r**2)
        E = MaterialProperties.from_name(
            self.cfg["inter_tank"]["stringer_material"]
        ).require("elastic_modulus")
        return E * I

    def get_area(self):
        r = self.cfg["vehicle"]["OMLD"] * 0.5
        self.lat_area = dist.uniform(geo.cylinder_lateral_area(r, self.length), self.n)
        self.surf_area = dist.uniform(geo.cylinder_surface_area(r, self.length), self.n)

    def get_MOI(self):
        r = self.cfg["vehicle"]["OMLD"] * 0.5
        self.cg = np.sum(self.mass * self.station) / np.sum(self.mass)
        self.Ixx = np.sum(self.mass * r**2)
        self.Iyy = np.sum(self.mass * (self.station - self.cg)**2)

# Legacy analytical aero / unused input container (inactive).
#     def get_CNa(self, M: float, alpha: float):
#         A_plan = self.cfg["vehicle"]["OMLD"] * self.length
#         self.CNa = dist.weighted(aero.body_CNa(M, alpha, A_plan, self.ref_area), self.lat_area)

    def get_thermal_oml_area(self) -> np.ndarray:
        return self.surf_area.copy()

    def get_thermal_shell_mass(self) -> np.ndarray:
        return self.shell_mass.copy()
