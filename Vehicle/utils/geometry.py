import numpy as np
from dataclasses import dataclass
from math import log, pi

def vk_profile(x: np.ndarray, L: float, R: float) -> np.ndarray:
    theta = np.arccos(np.clip(1 - 2 * x / L, -1.0, 1.0))
    return (R / np.sqrt(np.pi)) * np.sqrt(theta - 0.5 * np.sin(2 * theta))

def power_series_profile(x: np.ndarray, L: float, R: float, n: float) -> np.ndarray:
    xi = np.clip(x / L, 0.0, 1.0)
    return R * xi**n

def annulus_volume(r_o, r_i, L):
    return np.pi * (r_o**2 - r_i**2) * L

def annulus_second_moment(r_o, r_i):
    return 0.25 * np.pi * (r_o**4 - r_i**4)

def cylinder_lateral_area(r, L):
    return 2 * r * L

def cylinder_surface_area(r, L):
    return 2 * np.pi * r * L

## Dataclass for Rocket Aerodynamic Geometry ## 
@dataclass(init=False)
class RocketAeroGeometry:
    """Rocket geometry in SI units; angles are specified in degrees.

    The empty constructor supports incremental assembly with the ``set_*``
    methods. Values can instead be passed directly to the constructor.
    """

    root_chord: float | None
    tip_chord: float | None
    semi_span: float | None
    fin_count: int | None
    fin_cant: float
    fin_sweep: float
    nose_length: float | None
    nose_base_radius: float | None
    nose_shape: str | None
    body_length: float | None
    body_radius: float | None
    boattail_length: float | None
    boattail_aft_radius: float | None

    def __init__(self, root_chord=None, tip_chord=None, semi_span=None,
                 fin_count=None, fin_cant=0.0, *, fin_sweep=0.0,
                 nose_length=None, nose_base_radius=None, nose_shape=None,
                 body_length=None, body_radius=None, boattail_length=None,
                 boattail_aft_radius=None):
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.semi_span = semi_span
        self.fin_count = fin_count
        self.fin_cant = fin_cant
        self.fin_sweep = fin_sweep
        self.nose_length = nose_length
        self.nose_base_radius = nose_base_radius
        self.nose_shape = nose_shape
        self.body_length = body_length
        self.body_radius = body_radius
        self.boattail_length = boattail_length
        self.boattail_aft_radius = boattail_aft_radius

    def set_fin_size(self, root_chord, tip_chord, semi_span):
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.semi_span = semi_span
        return self

    def set_fin_geometry(self, fin_count, fin_cant=0.0, fin_sweep=0.0):
        self.fin_count = fin_count
        self.fin_cant = fin_cant
        self.fin_sweep = fin_sweep
        return self

    def set_nose_geometry(self, length, base_radius, shape="ogive"):
        self.nose_length = length
        self.nose_base_radius = base_radius
        self.nose_shape = shape
        return self

    def set_body_geometry(self, length, radius):
        self.body_length = length
        self.body_radius = radius
        return self

    def set_boattail_geometry(self, length, aft_radius):
        self.boattail_length, self.boattail_aft_radius = length, aft_radius
        return self

    setFinSize = set_fin_size
    setFinGeom = set_fin_geometry
    setNoseGeom = set_nose_geometry
    setBodyGeom = set_body_geometry
    setBoattailGeom = set_boattail_geometry

    @property
    def diameter(self):
        return 2.0 * self.body_radius

    @property
    def reference_area(self):
        return pi * self.body_radius**2

    @property
    def fin_area(self):
        return 0.5 * (self.root_chord + self.tip_chord) * self.semi_span

    @property
    def fin_aspect_ratio(self):
        return 2.0 * self.semi_span**2 / self.fin_area

    @property
    def total_length(self):
        return self.nose_length + self.body_length + self.boattail_length

    @property
    def roll_geometrical_constant(self):
        """Integral of ``c(xi) * xi**2`` for a trapezoidal fin."""
        cr, ct, span, radius = self.root_chord, self.tip_chord, self.semi_span, self.body_radius
        return span / 12.0 * ((cr + 3 * ct) * span**2
                            + 4 * (cr + 2 * ct) * span * radius
                            + 6 * (cr + ct) * radius**2)

    @property
    def roll_damping_interference_factor(self):
        """RocketPy/Barrowman trapezoidal fin-body interference factor K_d."""
        tau = (self.semi_span + self.body_radius) / self.body_radius
        lam = self.tip_chord / self.root_chord
        numerator = (tau - lam) / tau - (1 - lam) / (tau - 1) * log(tau)
        denominator = ((tau + 1) * (tau - lam) / 2
                       - (1 - lam) * (tau**3 - 1) / (3 * (tau - 1)))
        return 1.0 + numerator / denominator

    rootChord = property(lambda self: self.root_chord)
    tipChord = property(lambda self: self.tip_chord)
    semiSpan = property(lambda self: self.semi_span)
    finCount = property(lambda self: self.fin_count)
    finCant = property(lambda self: self.fin_cant)
