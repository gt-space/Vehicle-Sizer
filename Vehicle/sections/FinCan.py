import numpy as np
from ..Material import mp
from .Section import Section
from ..Engine import Engine
from ..utils import distribute as dist
from ..utils import geometry as geo

class FinCan(Section):

    def __init__(self, cfg: dict, engine: Engine):

        super().__init__(cfg)
        fin = cfg["fin_can"]
        self.length = float(fin["boattail_length"])
        self.set_grid()
        self.engine = engine
        self.span = float(fin["span"])
        self.root_chord = float(fin["root_chord"])
        self.tip_chord = float(fin["tip_chord"])
        self.sweep_fraction = float(fin["sweep_fraction"])
        self.fin_count = int(fin["fin_count"])
        self.fin_thickness = float(fin["fin_thickness"])
        self.boattail_aft_diameter = float(fin["boattail_aft_diameter"])
        self.wall_thickness = float(fin["boattail_wall_thickness"])
        self.wall_material = fin["material"]
        self.emissivity = 0.85
        if min(
            self.length,
            self.span,
            self.root_chord,
            self.tip_chord,
            self.fin_thickness,
            self.boattail_aft_diameter,
        ) <= 0.0:
            raise ValueError("Fin and boattail dimensions must be positive")
        if self.tip_chord > self.root_chord or self.root_chord > self.length:
            raise ValueError("Fin chords require tip <= root <= boattail length")
        if not 0.0 <= self.sweep_fraction <= 1.0:
            raise ValueError("Fin sweep_fraction must be between zero and one")
        if self.fin_count != 4:
            raise ValueError("The aerodynamic model requires exactly four fins")
        if self.wall_thickness <= 0 or 2 * self.wall_thickness >= min(self.boattail_aft_diameter, float(cfg["vehicle"]["OMLD"])):
            raise ValueError("Boattail wall must leave a positive internal diameter")
        self.radius = self.radius + self.local_centers / self.length * (self.boattail_aft_diameter / 2 - self.radius)
        self.inner_radius = self.radius - self.wall_thickness

    def get_mass(self):
        self.fin_shell_mass = self._get_fin_mass()
        self.boattail_shell_mass = self._get_boattail_mass_vector()
        hardware_mass = dist.uniform(self.fin_shell_mass, self.n)
        self.mass = (
            hardware_mass
            + self.boattail_shell_mass
        )

    def _get_fin_mass(self) -> float:
        V = self.fin_area * self.fin_thickness * self.fin_count
        mat = mp.get_material(self.cfg["fin_can"]["material"])
        rho = mat.get("density")
        m = rho * V
        return m


    def _get_boattail_mass_vector(self) -> np.ndarray:
        rho = mp.get_material(self.wall_material).get("density")
        return rho * geo.annulus_volume(self.radius, self.inner_radius, self.dx)

    def get_EI(self):
        E = mp.get_material(self.wall_material).get("elastic_modulus", T=300.0)
        self.EI = E * geo.annulus_second_moment(self.radius, self.inner_radius)

    def get_area(self):
        r_s_o = self.cfg["vehicle"]["OMLD"] * 0.5
        r_f_o = 0.5 * self.boattail_aft_diameter

        dr_dx = (r_f_o - r_s_o) / self.length
        ds = np.sqrt(self.dx**2 + (dr_dx * self.dx)**2)

        lat_body = 2.0 * self.radius * self.dx
        surf_body = 2.0 * np.pi * self.radius * ds

        fin_strip_area = self._fin_strip_area(self.local_centers)
        lat_fins = 2.0 * fin_strip_area
        surf_fins = 2.0 * self.fin_count * fin_strip_area

        self.lat_area = lat_body + lat_fins
        self.surf_area = surf_body + surf_fins
        self.surf_area_fins = surf_fins
        self.lat_area_fins = lat_fins
        self.lat_area_body = lat_body

    @property
    def fin_area(self) -> float:
        """Planform area of one trapezoidal fin."""

        return 0.5 * (self.root_chord + self.tip_chord) * self.span

    def _fin_strip_area(self, x_local: np.ndarray) -> np.ndarray:
        """Distribute one fin's planform area along its swept root chord."""

        x = x_local - (self.length - self.root_chord)
        y = (np.arange(512) + 0.5) * self.span / 512.0
        sweep = self.sweep_fraction * (self.root_chord - self.tip_chord)
        leading = sweep * y / self.span
        trailing = leading + self.root_chord + (
            self.tip_chord - self.root_chord
        ) * y / self.span
        height = self.span * np.mean(
            (x[:, None] >= leading) & (x[:, None] <= trailing),
            axis=1,
        )
        area = height * self.dx
        return area * (self.fin_area / np.sum(area))


    def get_thermal_shell_mass(self) -> np.ndarray:
        fin_weights = self.surf_area_fins
        if np.sum(fin_weights) == 0.0:
            fin_mass = dist.uniform(self.fin_shell_mass, self.n)
        else:
            fin_mass = self.fin_shell_mass * fin_weights / np.sum(fin_weights)
        return self.boattail_shell_mass + fin_mass
