"""RASAero II aerodynamic model for the team vehicle family: drag, normal
force, centre of pressure, and the normal-force and axial-force distributions
along the body.

    from dragmodel import DragModel
    m = DragModel()                       # loads dragmodel.h5 from this folder
    v = dict(omld=12, length=360, fineness=5, exit=9, boattail_aft=10.5, boattail_length=22,
             span=7.5, root=18, tip=6, sweep_fraction=0.75, thickness=0.375)      # inches

    m.cd(v, mach=2.0, alpha=5.0)                          # drag coefficient, power off
    m.cd(v, mach=2.0, alpha=5.0, power_on=True)
    m.cn(v, mach=2.0, alpha=5.0)                          # normal force coefficient
    m.cp(v, mach=2.0, alpha=5.0)                          # centre of pressure, in from the nose tip
    m.cp(v, mach=2.0, alpha=5.0, fins_on_boattail=True)
    m.cn_distribution(v, mach=2.0, alpha=5.0)             # dCN/dx along the body, with the parts
    m.cn_surface(v, alpha=5.0)                            # the same for every Mach on the grid
    m.ca_distribution(v, mach=2.0, alpha=5.0)             # dCA/dx along the body, running axial load, base point load
    m.ca_surface(v, alpha=5.0)                            # the same for every Mach on the grid
    m.cd_table(v), m.cn_table(v), m.cp_table(v)           # (73 Mach, 4 alpha) grids; m.mach, m.alpha

    python dragmodel.py            # self-test against RASAero II runs stored in the file

Every option takes nose ("vonkarman", "ogive", "conical") and finish
("10um", "20um", "30um"). Mach 0.1 to 10, alpha 0 to 15 deg. Sea level.
CD and CN are on the vehicle's own reference area, pi * D^2 / 4. Needs
numpy, scipy and h5py.

How it works: RASAero II's drag is a sum of body terms and fin terms and the
only variable both read is diameter, so drag is a body hypercube (a degree-5
polynomial on 4,096 runs per nose shape and finish) plus a fin hypercube (a
5-level grid per finish) minus a reference curve in diameter. Every
area-referenced quantity is stored on one fixed area, (D/D0)^2 times its
value, and divided back out here; older files without that flag are read as
plain coefficients. Normal force is built from
RASAero II's own per-part slopes and stations: the nose (with and without its
afterbody), the boattail, the fin set and its body carry-over, and the viscous
crossflow term. The body parts come from the same polynomial per nose shape, the fin
parts from one 5-level grid with the fin station moved to the vehicle's own
length, and the fin slopes scaled by a solved correction when the tube is
under 13 diameters. The distribution along the body spreads those lumps under stated
assumptions (see cn_distribution); it integrates to the same CN and CP.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from scipy.integrate import trapezoid

NOSES = ("vonkarman", "ogive", "conical")
FINISHES = ("10um", "20um", "30um")
VARIABLES = ("omld", "length", "fineness", "exit", "boattail_aft", "boattail_length",
             "span", "root", "tip", "sweep_fraction", "thickness")

#: Above Mach 1, RASAero II's fin and carry-over normal-force slopes grow once
#: the cylindrical tube between nose and boattail is shorter than this many
#: body diameters: no effect at 13, up to 2.5% at 11.5, 12% at 10.5, 23% at
#: 9.5, depending on fin shape and diameter together. The fin grid is solved on
#: the reference vehicle's tube, so files carry a solved correction (the
#: "shorttube" group) that parts_table applies. Only a file without it warns.
SHORT_TUBE_DIAMETERS = 12.0


class ShortTubeWarning(UserWarning):
    """CN and CP asked for a vehicle whose tube is short enough that RASAero II's
    supersonic fin slope depends on it, from a model file with no correction for it."""


ASSUMPTIONS = [
    "Nose alone (RASAero II's nose routine with no afterbody): load proportional to dS/dx of the actual profile, tilted to the nose-alone CP.",
    "Body lift (the rest of RASAero II's nose term, which grows with tube length): a smooth hump over the tube whose centroid reproduces RASAero II's nose moment.",
    "Boattail: load proportional to dS/dx (negative), tilted to RASAero II's boattail CP.",
    "Viscous crossflow: spread over the whole body in proportion to local diameter, per-part totals and centroids as RASAero II tabulates them.",
    "Fins and their body carry-over: a smooth hump over the fin root chord with the centroid at RASAero II's station for each term.",
    "Every curve integrates to the model's CN, and its first moment gives the model's CP.",
]
CA_ASSUMPTIONS = [
    "Body skin friction: over the whole wetted body in proportion to local circumference times a turbulent local skin-friction law, x^-0.2 from the nose tip.",
    "Nose wave drag: over the nose in proportion to dS/dx times sin^2 of the local surface angle (Newtonian pressure), so it gathers where the nose is steepest.",
    "Form drag (subsonic): over the nose and the boattail in proportion to |dS/dx|, as the pressure drag of the forebody and afterbody.",
    "Boattail wave drag: over the boattail in proportion to |dS/dx|.",
    "Base drag, and the power-on base credit: a point load on the base plane.",
    "Fins (profile, friction, wave, interference, edge): over the fin root chord in proportion to one fin's planform area per unit length.",
    "Above alpha 0 RASAero II gives no split: the alpha-0 split is scaled to the model's CA at that alpha.",
    "Inside RASAero II's transonic fairing (Mach 0.9 to 1.05) it gives no split either: the Mach 0.9 and 1.05 splits are blended linearly in Mach.",
    "Every curve integrates to the model's CA; running_ca is the axial force from the nose tip back to each station.",
]


def _chebyshev_features(U, powers):
    Z = 2.0 * np.asarray(U, float) - 1.0
    n, d = Z.shape
    deg = int(powers.max())
    T = np.empty((deg + 1, n, d)); T[0] = 1.0
    if deg >= 1:
        T[1] = Z
    for k in range(2, deg + 1):
        T[k] = 2.0 * Z * T[k - 1] - T[k - 2]
    F = np.ones((n, len(powers)))
    for j, row in enumerate(powers):
        for axis, p in enumerate(row):
            if p:
                F[:, j] *= T[p, :, axis]
    return F


def _trap(y, x):
    return float(trapezoid(y, x))


class DragModel:
    def __init__(self, path: str | Path | None = None):
        path = Path(path) if path else Path(__file__).with_name("dragmodel.h5")
        self.path = path
        with h5py.File(path, "r") as f:
            self.mach = f["mach"][()]; self.alpha = f["alpha"][()]
            self.space = json.loads(f.attrs["space"])
            self.ranges = {a["name"]: (a["low"], a["high"]) for a in self.space["axes"]}
            # The boattail's own floor. In the current space it is a plain constant
            # and needs no special handling; in older spaces its low end was the
            # *exit*, and the body block was solved with the exit pinned at the
            # space's floor, so the block coordinate was measured from there rather
            # than from the vehicle's own exit. Stored with the model either way.
            bt_low = self.ranges["boattail_aft"][0]
            self.boattail_floor = (float(bt_low) if not isinstance(bt_low, str)
                                   else float(f.attrs.get("boattail_floor", 8.0)))
            self.info = {k: str(f.attrs[k]) for k in ("created", "solver", "notes") if k in f.attrs}
            self.reference = json.loads(f.attrs["reference_vehicle"]) if "reference_vehicle" in f.attrs else None
            # Every area-referenced quantity may be stored on one fixed area,
            # (D/D0)^2 times its value, which takes the 1/D^2 of the vehicle's own
            # reference area out of the fits. Older files store plain coefficients.
            self.fixed_area = str(f.attrs.get("cd_scaling", "none")) == "fixed_area"
            self.d0 = float(f.attrs.get("d0", 1.0))
            self.ladder = f["power_on/ladder"][()]
            self.body = {k: {"coef": g["coef"][()], "powers": g["powers"][()], "axes": json.loads(g.attrs["axes"])} for k, g in f["body"].items()}
            self.fins, self.ref = {}, {}
            for finish, g in f["fins"].items():
                axes = json.loads(g.attrs["axes"]); L = int(g.attrs["levels"])
                # fixed-area files hold the fin increment over the reference curve
                table = g["delta"][()] if "delta" in g else g["cd"][()]
                self.fins[finish] = {"axes": axes, "interp": RegularGridInterpolator([np.linspace(0, 1, L)] * len(axes), table, method="linear")}
            for finish, g in f["ref"].items():
                self.ref[finish] = PchipInterpolator(g["u"][()], g["cd"][()], axis=0)
            self.bodyparts = {}
            if "bodyparts" in f:
                for nose, g in f["bodyparts"].items():
                    self.bodyparts[nose] = {"coef": g["coef"][()], "powers": g["powers"][()], "axes": json.loads(g.attrs["axes"]), "columns": json.loads(g.attrs["columns"])}
            self.finparts = None
            if "finparts" in f:
                g = f["finparts/grid"]; axes = json.loads(g.attrs["axes"]); L = int(g.attrs["levels"])
                self.finparts = {"axes": axes, "columns": json.loads(g.attrs["columns"]),
                                 "interp": RegularGridInterpolator([np.linspace(0, 1, L)] * len(axes), g["vals"][()], method="linear"),
                                 "ref_length": float(g.attrs["ref_length"]), "ref_boattail_length": float(g.attrs["ref_boattail_length"])}
            # Short-tube correction for the fin slopes: a factor over (tube/D, D, span,
            # root, tip, sweep) per Mach, exactly 1 at the top tube node and above.
            self.shorttube = None
            if "shorttube" in f:
                g = f["shorttube"]; axes = json.loads(g.attrs["axes"]); nodes = [g[a][()] for a in axes]
                self.shorttube = {"axes": axes, "columns": json.loads(g.attrs["columns"]),
                                  "lo": np.array([n[0] for n in nodes]), "hi": np.array([n[-1] for n in nodes]),
                                  "top": float(nodes[0][-1]),
                                  "interp": RegularGridInterpolator(nodes, g["k"][()], method="linear")}
            # RASAero II's alpha-0 drag split by where it acts, for the axial-force distribution
            self.casplit = None
            if "casplit" in f:
                g = f["casplit"]
                self.casplit = {"classes": json.loads(g.attrs["classes"]), "fit_mask": np.asarray(g.attrs["fit_mask"]).astype(bool),
                                "columns": json.loads(g.attrs["breakdown_columns"]),
                                "fits": {k: {"coef": sg["coef"][()], "powers": sg["powers"][()], "axes": json.loads(sg.attrs["axes"])}
                                         for k, sg in g.items()}}

    # ------------------------------------------------------------ coordinates and geometry
    def check(self, v: dict) -> None:
        """Raise ValueError if the vehicle is outside the ranges the model was built on."""
        missing = [k for k in VARIABLES if k not in v]
        if missing:
            raise ValueError(f"vehicle is missing {missing}")
        if v["exit"] > v["boattail_aft"] + 1e-9:
            raise ValueError(f"exit = {v['exit']} is larger than boattail_aft = {v['boattail_aft']}; the nozzle must fit inside the boattail exit plane")
        if v["boattail_aft"] > v["omld"] + 1e-9:
            raise ValueError(f"boattail_aft = {v['boattail_aft']} is larger than omld = {v['omld']}")
        for name in VARIABLES:
            lo, hi = self.ranges[name]
            lo = v[lo] if isinstance(lo, str) else lo
            hi = v[hi] if isinstance(hi, str) else hi
            if not (lo - 1e-9 <= v[name] <= hi + 1e-9):
                raise ValueError(f"{name} = {v[name]} is outside [{lo}, {hi}]")

    def _scale(self, v: dict) -> float:
        """(D/D0)^2 for a fixed-area file, which the reader divides back out; 1 otherwise."""
        return (v["omld"] / self.d0) ** 2 if self.fixed_area else 1.0

    def _unit(self, v: dict, axes) -> np.ndarray:
        u = []
        for name in axes:
            lo, hi = self.ranges[name]
            if name == "boattail_aft":
                lo = self.boattail_floor          # never the vehicle's own exit
            lo = v[lo] if isinstance(lo, str) else lo
            hi = v[hi] if isinstance(hi, str) else hi
            u.append((v[name] - lo) / (hi - lo))
        return np.clip(np.array(u, float), 0.0, 1.0)

    @staticmethod
    def tube_diameters(v: dict) -> float:
        """Length of the cylindrical tube between nose and boattail, in body diameters.
        Below SHORT_TUBE_DIAMETERS the model's supersonic CN and CP are not trusted."""
        return (v["length"] - v["fineness"] * v["omld"] - v["boattail_length"]) / v["omld"]

    @staticmethod
    def stations(v: dict) -> dict:
        """Nose length, tube start and length, boattail start, fin root leading edge (tube and boattail placements), total length."""
        Ln = v["fineness"] * v["omld"]; Lb = v["boattail_length"]; Lt = v["length"] - Ln - Lb
        return {"nose_length": Ln, "tube_x0": Ln, "tube_length": Lt, "boattail_x0": Ln + Lt, "boattail_length": Lb,
                "fin_x0": Ln + Lt - v["root"], "fin_x0_boattail": v["length"] - v["root"], "length": v["length"]}

    @staticmethod
    def nose_radius(nose: str, L: float, R: float, x):
        x = np.clip(np.asarray(x, float), 0.0, L)
        if nose == "conical":
            return R * x / L
        if nose == "ogive":
            rho = (R ** 2 + L ** 2) / (2.0 * R)
            return np.sqrt(np.maximum(rho ** 2 - (L - x) ** 2, 0.0)) + R - rho
        theta = np.arccos(np.clip(1.0 - 2.0 * x / L, -1.0, 1.0))
        return R / np.sqrt(np.pi) * np.sqrt(np.maximum(theta - np.sin(2.0 * theta) / 2.0, 0.0))

    def diameter_profile(self, v: dict, nose: str, x):
        s = self.stations(v); D = v["omld"]; d = np.zeros_like(x, dtype=float)
        m = x <= s["nose_length"]; d[m] = 2.0 * self.nose_radius(nose, s["nose_length"], D / 2.0, x[m])
        m = (x > s["tube_x0"]) & (x <= s["boattail_x0"]); d[m] = D
        m = x > s["boattail_x0"]; d[m] = D + (v["boattail_aft"] - D) * np.clip((x[m] - s["boattail_x0"]) / s["boattail_length"], 0, 1)
        return d

    def planform(self, v: dict, nose: str):
        """RASAero II's per-part planform areas and centroids (nose, tube, boattail) and the whole-body centroid."""
        s = self.stations(v); D = v["omld"]; Ln, Lt, Lb = s["nose_length"], s["tube_length"], s["boattail_length"]; da = v["boattail_aft"]
        if nose == "conical":
            parts = [("nose", 0.0, Ln, 0.5 * Ln * D, 2.0 / 3.0 * Ln)]
        else:
            parts = [("nose", 0.0, Ln, 2.0 / 3.0 * Ln * D, 0.625 * Ln)]
        parts.append(("tube", s["tube_x0"], Lt, Lt * D, s["tube_x0"] + 0.5 * Lt))
        parts.append(("boattail", s["boattail_x0"], Lb, Lb * (D + da) / 2.0, s["boattail_x0"] + Lb * (D + 2.0 * da) / (3.0 * (D + da))))
        area = sum(p[3] for p in parts)
        return parts, sum(p[3] * p[4] for p in parts) / area

    # ------------------------------------------------------------ drag
    def cd_table(self, v: dict, nose: str = "vonkarman", finish: str = "10um", power_on: bool = False) -> np.ndarray:
        """Wind-axis CD on the (73 Mach, 4 alpha) grid."""
        self.check(v)
        key = f"{nose}_{finish}"
        if key not in self.body or finish not in self.fins:
            raise ValueError(f"no model for nose {nose!r} with finish {finish!r}")
        b = self.body[key]
        body = (_chebyshev_features(self._unit(v, b["axes"])[None, :], b["powers"]) @ b["coef"]).reshape(len(self.mach), len(self.alpha))
        f = self.fins[finish]
        fins = f["interp"](self._unit(v, f["axes"])[None, :])[0]
        if self.fixed_area:
            # body and the fin increment are both on the fixed area; the reference
            # curve is already folded into the increment
            cd = (body + fins) / self._scale(v)
        else:
            cd = body + fins - self.ref[finish](self._unit(v, ["omld"])[0])
        if power_on:
            cd = cd - self.ladder[:, None] * (v["exit"] / v["omld"]) ** 2 / np.cos(np.radians(self.alpha))[None, :]
        return cd

    # ------------------------------------------------------------ normal force
    def parts_table(self, v: dict, nose: str = "vonkarman") -> dict:
        """RASAero II's per-part normal-force slopes (per rad) and stations (in) per Mach, and the viscous CN per Mach and alpha."""
        if not self.bodyparts or self.finparts is None:
            raise ValueError("this model file carries drag only; rebuild with the normal-force pieces")
        self.check(v)
        td = self.tube_diameters(v); st = self.shorttube
        if td < SHORT_TUBE_DIAMETERS and not (st is not None and td >= st["lo"][0] - 1e-9):
            # Only a file without the short-tube correction (or a tube shorter than
            # it was solved for) gets here. One fixed message, so the default
            # filter prints it once rather than once per vehicle inside an optimizer.
            warnings.warn(ShortTubeWarning(
                f"tube shorter than {SHORT_TUBE_DIAMETERS:g} diameters and this model file has no short-tube "
                f"correction for it: supersonic CN is under-predicted (up to about 23% at 9.5 diameters) and "
                f"CP can be far off. Drag at alpha 0 is unaffected. Check a vehicle with "
                f"DragModel.tube_diameters(v)."), stacklevel=3)
        b = self.bodyparts[nose]; M = len(self.mach)
        y = (_chebyshev_features(self._unit(v, b["axes"])[None, :], b["powers"]) @ b["coef"])[0]
        nb = len(b["columns"])
        s = self._scale(v)                       # slopes and viscous CN are on A_ref; stations are lengths
        out = {c: y[:M * nb].reshape(M, nb)[:, i] / (s if c.endswith("_cna") else 1.0) for i, c in enumerate(b["columns"])}
        out["cn_viscous"] = y[M * nb:].reshape(M, len(self.alpha)) / s
        f = self.finparts
        vals = f["interp"](self._unit(v, f["axes"])[None, :])[0]                       # (73, 6)
        vals = vals / np.array([s if c.endswith("_cna") else 1.0 for c in f["columns"]])[None, :]
        if st is not None and td < st["top"]:
            # The fin grid is solved on the reference body's tube; below 13 diameters
            # RASAero II's supersonic fin and carry-over slopes grow, by a factor
            # that depends on the fin. Scale them by the solved correction.
            q = np.clip(np.array([td if a == "tube_d" else v[a] for a in st["axes"]], float), st["lo"], st["hi"])
            k = st["interp"](q[None, :])[0]                                           # (73, 2)
            for j, c in enumerate(st["columns"]):
                vals[:, f["columns"].index(c)] *= k[:, j]
        s = self.stations(v)
        shift = (v["length"] - v["boattail_length"]) - (f["ref_length"] - f["ref_boattail_length"])
        shift_bt = v["length"] - f["ref_length"]
        for i, c in enumerate(f["columns"]):
            col = vals[:, i]
            if c in ("fin_cp", "carry_cp"):
                col = col + shift
            elif c in ("fin_cp_bt", "carry_cp_bt"):
                col = col + shift_bt
            out[c] = col
        out["stations"] = s
        return out

    def cn_table(self, v: dict, nose: str = "vonkarman") -> np.ndarray:
        """CN on the (73 Mach, 4 alpha) grid."""
        p = self.parts_table(v, nose)
        cna = p["nose_cna"] + p["btail_cna"] + p["fin_cna"] + p["carry_cna"]
        return cna[:, None] * np.radians(self.alpha)[None, :] + p["cn_viscous"]

    def cp_table(self, v: dict, nose: str = "vonkarman", fins_on_boattail: bool = False) -> np.ndarray:
        """CP in inches from the nose tip on the (73 Mach, 4 alpha) grid. The alpha-0 column is the potential-flow limit."""
        p = self.parts_table(v, nose)
        fin_cp, carry_cp = (p["fin_cp_bt"], p["carry_cp_bt"]) if fins_on_boattail else (p["fin_cp"], p["carry_cp"])
        cna = p["nose_cna"] + p["btail_cna"] + p["fin_cna"] + p["carry_cna"]
        m0 = p["nose_cna"] * p["nose_cp"] + p["btail_cna"] * p["btail_cp"] + p["fin_cna"] * fin_cp + p["carry_cna"] * carry_cp
        _, centroid = self.planform(v, nose)
        a = np.radians(self.alpha)
        cn = cna[:, None] * a[None, :] + p["cn_viscous"]
        mom = m0[:, None] * a[None, :] + p["cn_viscous"] * centroid
        cp = np.empty_like(cn); cp[:, 0] = m0 / cna; cp[:, 1:] = mom[:, 1:] / cn[:, 1:]
        # RASAero II blends CP itself through the transonic fairing, linearly in Mach between its
        # Mach 0.9 and 1.05 solutions; do the same rather than take the ratio of blended parts.
        lo, hi = int(np.argmin(np.abs(self.mach - 0.9))), int(np.argmin(np.abs(self.mach - 1.05)))
        for k in range(len(self.mach)):
            if 0.9 < self.mach[k] < 1.05:
                u = (self.mach[k] - 0.9) / 0.15
                cp[k] = cp[lo] + (cp[hi] - cp[lo]) * u
        return cp

    # ------------------------------------------------------------ point evaluation
    def _at(self, table, mach, alpha):
        mach = np.asarray(mach, float); alpha = np.asarray(alpha, float)
        if np.any(mach < self.mach[0]) or np.any(mach > self.mach[-1]) or np.any(alpha < 0) or np.any(alpha > self.alpha[-1]):
            raise ValueError(f"Mach must be in [{self.mach[0]}, {self.mach[-1]}] and alpha in [0, {self.alpha[-1]}]")
        at_alpha = PchipInterpolator(self.alpha, table, axis=1)(np.atleast_1d(alpha))
        out = np.array([np.interp(np.atleast_1d(mach), self.mach, at_alpha[:, k]) for k in range(at_alpha.shape[1])])
        out = np.broadcast_to(out, np.broadcast(np.atleast_1d(alpha)[:, None], np.atleast_1d(mach)[None, :]).shape)
        return float(out[0, 0]) if mach.ndim == 0 and alpha.ndim == 0 else np.squeeze(out)

    def cd(self, v, mach, alpha=0.0, nose="vonkarman", finish="10um", power_on=False):
        """CD at any Mach in [0.1, 10] and alpha in [0, 15] deg. Scalars or arrays."""
        return self._at(self.cd_table(v, nose, finish, power_on), mach, alpha)

    def cn(self, v, mach, alpha=0.0, nose="vonkarman", finish="10um"):
        """CN at any Mach and alpha. Finish does not enter normal force; accepted for symmetry."""
        return self._at(self.cn_table(v, nose), mach, alpha)

    def cp(self, v, mach, alpha=0.0, nose="vonkarman", finish="10um", fins_on_boattail=False):
        """Centre of pressure, inches from the nose tip, at any Mach and alpha."""
        return self._at(self.cp_table(v, nose, fins_on_boattail), mach, alpha)

    def ca(self, v, mach, alpha=0.0, nose="vonkarman", finish="10um", power_on=False):
        """Axial force coefficient, body axes: (CD_wind - CN sin alpha) / cos alpha."""
        a = np.radians(np.asarray(alpha, float))
        return (self.cd(v, mach, alpha, nose, finish, power_on) - self.cn(v, mach, alpha, nose) * np.sin(a)) / np.cos(a)

    # ------------------------------------------------------------ distribution along the body
    @staticmethod
    def _normalise(x, w, total):
        area = _trap(w, x)
        return w * (total / area) if area else np.zeros_like(w)

    @staticmethod
    def _tilt(x, w, target):
        W = _trap(w, x)
        if W == 0:
            return w
        xbar = _trap(w * x, x) / W; var = _trap(w * (x - xbar) ** 2, x) / W
        if var <= 0:
            return w
        k = (target - xbar) / var
        lo, hi = x[w > 0].min(), x[w > 0].max()
        kmax = 1.0 / max(xbar - lo, 1e-9) if k < 0 else 1.0 / max(hi - xbar, 1e-9)
        w2 = w * (1.0 + np.clip(k, -kmax, kmax) * (x - xbar))
        return w2 * (W / _trap(w2, x))

    @staticmethod
    def _hump(x, a, b, centroid):
        t = np.clip((x - a) / (b - a), 0.0, 1.0); m = np.clip((centroid - a) / (b - a), 0.05, 0.95)
        if m >= 0.5:
            q = 1.0; p = (3.0 * m - 1.0) / (1.0 - m)
        else:
            p = 1.0; q = (2.0 - 3.0 * m) / m
        w = t ** p * (1.0 - t) ** q; w[(x < a) | (x > b)] = 0.0
        return w

    def cn_distribution(self, v: dict, mach: float, alpha: float, nose: str = "vonkarman", finish: str = "10um", n: int = 800, fins_on_boattail: bool = False) -> dict:
        """dCN/dx along the body (per inch, on the reference area) at one Mach and alpha, split by component.

        Returns x, dcn_dx, parts, running_cn, cn, cp. See ASSUMPTIONS for how each lump is spread.
        """
        if 0.9 < mach < 1.05:
            # RASAero II's transonic fairing: blend the Mach 0.9 and 1.05 distributions linearly in Mach, as RASAero II blends its totals
            lo = self.cn_distribution(v, 0.9, alpha, nose, finish, n, fins_on_boattail); hi = self.cn_distribution(v, 1.05, alpha, nose, finish, n, fins_on_boattail)
            u = (float(mach) - 0.9) / 0.15; x = lo["x"]
            keys = set(lo["parts"]) | set(hi["parts"])
            parts = {k: (1 - u) * lo["parts"].get(k, 0.0) + u * hi["parts"].get(k, 0.0) for k in keys}
            total = sum(parts.values()); cn = _trap(total, x)
            return {"x": x, "diameter": lo["diameter"], "dcn_dx": total, "parts": parts, "cn": cn, "cp": (_trap(total * x, x) / cn) if cn else float("nan"),
                    "running_cn": np.concatenate([[0.0], np.cumsum(0.5 * (total[1:] + total[:-1]) * np.diff(x))]), "mach": float(mach), "alpha": float(alpha)}
        p = self.parts_table(v, nose)
        s = p["stations"]; a = np.radians(float(alpha)); M = float(mach)
        col = lambda name: float(np.interp(M, self.mach, p[name]))
        x = np.linspace(0.0, s["length"], n); D = self.diameter_profile(v, nose, x); S = np.pi * (D / 2.0) ** 2; dSdx = np.gradient(S, x)
        parts = {}
        # nose alone, and the afterbody lift that RASAero II folds into its nose term
        seg = x <= s["nose_length"]
        w = np.zeros_like(x); w[seg] = np.abs(dSdx[seg])
        parts["nose"] = self._tilt(x, self._normalise(x, w, col("nose0_cna") * a), col("nose0_cp"))
        cn_rem = (col("nose_cna") - col("nose0_cna")) * a
        if abs(cn_rem) > 1e-12:
            x_rem = (col("nose_cna") * col("nose_cp") - col("nose0_cna") * col("nose0_cp")) / (col("nose_cna") - col("nose0_cna"))
            a0, b0 = s["tube_x0"], s["tube_x0"] + s["tube_length"]
            parts["body lift (afterbody)"] = self._normalise(x, self._hump(x, a0, b0, x_rem), cn_rem)
        # boattail
        cn_bt = col("btail_cna") * a
        if cn_bt != 0.0:
            seg = x >= s["boattail_x0"]; w = np.zeros_like(x); w[seg] = np.abs(dSdx[seg])
            parts["boattail"] = self._tilt(x, self._normalise(x, w, abs(cn_bt)), col("btail_cp")) * np.sign(cn_bt)
        # viscous crossflow, RASAero II's per-part planform shares, spread inside each part by local diameter
        visc = float(PchipInterpolator(self.alpha, np.array([np.interp(M, self.mach, p["cn_viscous"][:, j]) for j in range(len(self.alpha))]))(alpha))
        if visc:
            plan, _ = self.planform(v, nose); total_area = sum(q[3] for q in plan); w = np.zeros_like(x)
            for name, x0, length, area, centroid in plan:
                seg = (x >= x0) & (x <= x0 + length); wp = np.zeros_like(x); wp[seg] = D[seg]
                w += self._tilt(x, self._normalise(x, wp, visc * area / total_area), centroid)
            parts["viscous crossflow"] = w
        # fins and their carry-over on the body
        x0 = s["fin_x0_boattail"] if fins_on_boattail else s["fin_x0"]; a0, b0 = x0, x0 + v["root"]
        fin_cp, carry_cp = ("fin_cp_bt", "carry_cp_bt") if fins_on_boattail else ("fin_cp", "carry_cp")
        for lab, cna, cpn in (("fins", "fin_cna", fin_cp), ("fin carry-over on body", "carry_cna", carry_cp)):
            cn_part = col(cna) * a
            if cn_part:
                parts[lab] = self._normalise(x, self._hump(x, a0, b0, col(cpn)), cn_part)
        total = sum(parts.values()) if parts else np.zeros_like(x)
        cn = _trap(total, x)
        return {"x": x, "diameter": D, "dcn_dx": total, "parts": parts, "cn": cn, "cp": (_trap(total * x, x) / cn) if cn else float("nan"),
                "running_cn": np.concatenate([[0.0], np.cumsum(0.5 * (total[1:] + total[:-1]) * np.diff(x))]), "mach": M, "alpha": float(alpha)}

    def cn_surface(self, v: dict, alpha: float, machs=None, nose: str = "vonkarman", finish: str = "10um", n: int = 800, fins_on_boattail: bool = False) -> dict:
        """cn_distribution at every Mach on the grid (or the Machs given): dcn_dx is (Mach, station)."""
        machs = np.asarray(self.mach if machs is None else machs, float)
        rows = [self.cn_distribution(v, float(m), alpha, nose, finish, n, fins_on_boattail) for m in machs]
        keys = sorted({k for r in rows for k in r["parts"]})
        return {"x": rows[0]["x"], "mach": machs, "alpha": float(alpha), "dcn_dx": np.array([r["dcn_dx"] for r in rows]),
                "parts": {k: np.array([r["parts"].get(k, np.zeros(n)) for r in rows]) for k in keys},
                "running_cn": np.array([r["running_cn"] for r in rows]), "cn": np.array([r["cn"] for r in rows]), "cp": np.array([r["cp"] for r in rows])}

    # ------------------------------------------------------------ axial force along the body
    def ca_split_table(self, v: dict, nose: str = "vonkarman", finish: str = "10um") -> dict:
        """RASAero II's alpha-0 drag split by where it acts, per Mach on the grid.

        {class: (73,)} for friction, form, nose wave, base, boattail wave and fins;
        the classes sum to the model's CD at alpha 0, power off, at every Mach.
        Inside the transonic fairing, where RASAero II records no split, the
        Mach 0.9 and 1.05 splits are blended linearly in Mach.
        """
        if self.casplit is None:
            raise ValueError("this model file carries no axial split; rebuild it with build_dragmodel.py")
        self.check(v)
        c = self.casplit["fits"][f"{nose}_{finish}"]; classes = self.casplit["classes"]; mask = self.casplit["fit_mask"]
        s = self._scale(v)
        y = (_chebyshev_features(self._unit(v, c["axes"])[None, :], c["powers"]) @ c["coef"])[0].reshape(int(mask.sum()), len(classes)) / s
        # the fins' share is the reference fins' part, carried by the body fit, plus
        # the fin increment the drag model already carries for this vehicle's fins
        f = self.fins[finish]
        fins = f["interp"](self._unit(v, f["axes"])[None, :])[0][:, 0]
        delta = fins / s if self.fixed_area else fins - self.ref[finish](self._unit(v, ["omld"])[0])[:, 0]
        tab = np.zeros((len(self.mach), len(classes))); tab[mask] = y
        tab[mask, classes.index("fins")] += delta[mask]
        frac = np.zeros_like(tab); frac[mask] = tab[mask] / tab[mask].sum(axis=1, keepdims=True)
        lo = int(np.flatnonzero(mask & (self.mach <= 0.9))[-1]); hi = int(np.flatnonzero(mask & (self.mach >= 1.05))[0])
        for i in np.flatnonzero(~mask):
            w = (self.mach[i] - self.mach[lo]) / (self.mach[hi] - self.mach[lo])
            frac[i] = (1.0 - w) * frac[lo] + w * frac[hi]
        cd0 = self.cd_table(v, nose, finish)[:, 0]
        return {k: frac[:, j] * cd0 for j, k in enumerate(classes)}

    def ca_distribution(self, v: dict, mach: float, alpha: float = 0.0, nose: str = "vonkarman", finish: str = "10um",
                        n: int = 800, fins_on_boattail: bool = False, power_on: bool = False) -> dict:
        """dCA/dx along the body (per inch, on the reference area) at one Mach and alpha, split by where each drag component acts.

        Returns x, diameter, dca_dx, parts, point_loads, running_ca, ca. running_ca is the axial force coefficient
        from the nose tip back to each station. The base acts as a point load on the base plane: it is listed in
        point_loads as (station, load), and it also sits in dca_dx and parts["base"] as a spike on the last station
        whose integral is exactly that load, so running_ca reaches ca at the base. See CA_ASSUMPTIONS.
        """
        M = float(mach)
        split = self.ca_split_table(v, nose, finish)
        share = {k: float(np.interp(M, self.mach, val)) for k, val in split.items()}
        total_share = sum(share.values())
        ca_off = float(self.ca(v, M, alpha, nose, finish, False))
        load = {k: ca_off * val / total_share for k, val in share.items()}
        if power_on:
            load["base"] += float(self.ca(v, M, alpha, nose, finish, True)) - ca_off      # the credit acts on the base
        s = self.stations(v)
        x = np.linspace(0.0, s["length"], n); dx = x[1] - x[0]
        D = self.diameter_profile(v, nose, x); r = D / 2.0
        dSdx = np.abs(np.gradient(np.pi * r ** 2, x)); drdx = np.gradient(r, x)
        nose_seg = x <= s["nose_length"]; bt_seg = x >= s["boattail_x0"]

        def on(seg, w):
            w = np.where(seg, w, 0.0)
            return w if _trap(w, x) > 0 else np.where(seg, 1.0, 0.0)       # flat, if the shape has no slope to weight by

        weights = {
            "friction": np.pi * D * np.sqrt(1.0 + drdx ** 2) * np.maximum(x, 0.5 * dx) ** -0.2,
            "form": on(nose_seg | bt_seg, dSdx),
            "nose wave": on(nose_seg, dSdx * drdx ** 2 / (1.0 + drdx ** 2)),
            "boattail wave": on(bt_seg, dSdx),
        }
        # one fin's planform area per unit length: up the swept leading edge, along the tip, down the trailing edge
        x0 = s["fin_x0_boattail"] if fins_on_boattail else s["fin_x0"]
        sweep = v["sweep_fraction"] * (v["root"] - v["tip"])
        te = max(x0 + v["root"], x0 + sweep + v["tip"] + 1e-6)
        weights["fins"] = np.interp(x, [x0, x0 + sweep, x0 + sweep + v["tip"], te], [0.0, v["span"], v["span"], 0.0], left=0.0, right=0.0)
        parts = {k: self._normalise(x, w, load[k]) for k, w in weights.items() if load.get(k, 0.0) != 0.0}
        base = np.zeros_like(x); base[-1] = 2.0 * load["base"] / dx            # a trapezoid rule reads this as exactly load["base"]
        parts["base"] = base
        total = sum(parts.values())
        running = np.concatenate([[0.0], np.cumsum(0.5 * (total[1:] + total[:-1]) * np.diff(x))])
        return {"x": x, "diameter": D, "dca_dx": total, "parts": parts, "point_loads": {"base": (float(s["length"]), load["base"])},
                "running_ca": running, "ca": _trap(total, x), "mach": M, "alpha": float(alpha), "power_on": bool(power_on)}

    def ca_surface(self, v: dict, alpha: float = 0.0, machs=None, nose: str = "vonkarman", finish: str = "10um", n: int = 800,
                   fins_on_boattail: bool = False, power_on: bool = False) -> dict:
        """ca_distribution at every Mach on the grid (or the Machs given): dca_dx and running_ca are (Mach, station)."""
        machs = np.asarray(self.mach if machs is None else machs, float)
        rows = [self.ca_distribution(v, float(m), alpha, nose, finish, n, fins_on_boattail, power_on) for m in machs]
        keys = sorted({k for r in rows for k in r["parts"]})
        return {"x": rows[0]["x"], "mach": machs, "alpha": float(alpha), "power_on": bool(power_on), "dca_dx": np.array([r["dca_dx"] for r in rows]),
                "parts": {k: np.array([r["parts"].get(k, np.zeros(n)) for r in rows]) for k in keys},
                "base_load": np.array([r["point_loads"]["base"][1] for r in rows]),
                "running_ca": np.array([r["running_ca"] for r in rows]), "ca": np.array([r["ca"] for r in rows])}

    # ------------------------------------------------------------ self-test
    def verify(self, log=print) -> dict:
        """Compare the model with RASAero II on the held-out vehicles stored in the file."""
        out = {}
        with h5py.File(self.path, "r") as f:
            for key in f["verify"]:
                g = f[f"verify/{key}"]; names = json.loads(g["x"].attrs["names"]); X = g["x"][()]
                nose, finish = key.rsplit("_", 1)
                have_nf = bool(self.bodyparts) and "cn" in g
                have_split = self.casplit is not None and "breakdown" in g
                e = {k: [] for k in ("off", "on", "cn", "cp", "cp_bt", "dist", "split", "ca_dist")}
                for i in range(len(X)):
                    v = {nm: float(X[i, k]) for k, nm in enumerate(names)}
                    p = self.cd_table(v, nose, finish); e["off"].append(np.max(np.abs(p - g["cd_off"][i]) / g["cd_off"][i], axis=0))
                    p = self.cd_table(v, nose, finish, power_on=True); e["on"].append(np.max(np.abs(p - g["cd_on"][i]) / g["cd_on"][i], axis=0))
                    if have_split:
                        # each class's share of CD against RASAero II's own breakdown, outside the fairing
                        mask = self.casplit["fit_mask"]; classes = self.casplit["classes"]
                        sp = self.ca_split_table(v, nose, finish); tb = g["breakdown"][i][mask]
                        truth = np.stack([tb[:, self.casplit["columns"][k]].sum(axis=-1) for k in classes], axis=-1)
                        mine = np.stack([sp[k][mask] for k in classes], axis=-1)
                        e["split"].append(100 * np.max(np.abs(mine / mine.sum(1, keepdims=True) - truth / truth.sum(1, keepdims=True))))
                        if i < 5:
                            d = self.ca_distribution(v, 2.0, 5.0, nose, finish, power_on=True)
                            e["ca_dist"].append(abs(d["ca"] - self.ca(v, 2.0, 5.0, nose, finish, True)) / abs(d["ca"]))
                    if have_nf:
                        cn = self.cn_table(v, nose); tr = g["cn"][i]
                        e["cn"].append(np.max(np.abs(cn[:, 1:] - tr[:, 1:]) / np.abs(tr[:, 1:]), axis=0))
                        cp = self.cp_table(v, nose); tr = g["cp"][i]; e["cp"].append(np.max(np.abs(cp - tr), axis=0))
                        cp = self.cp_table(v, nose, fins_on_boattail=True); tr = g["cp_boattail"][i]; e["cp_bt"].append(np.max(np.abs(cp - tr), axis=0))
                        if i < 5:
                            d = self.cn_distribution(v, 2.0, 10.0, nose, finish)
                            e["dist"].append(abs(d["cn"] - self.cn(v, 2.0, 10.0, nose)) / abs(d["cn"]))
                off, on = 100 * np.array(e["off"]), 100 * np.array(e["on"])
                r = {"n": len(X), "cd_off_median": float(np.median(off[:, 0])), "cd_off_p95": float(np.percentile(off[:, 0], 95)), "cd_off_max": float(off[:, 0].max()),
                     "cd_off_a15_p95": float(np.percentile(off[:, 3], 95)), "cd_on_p95": float(np.percentile(on[:, 0], 95)), "cd_on_max": float(on[:, 0].max())}
                line = f"{key:16s} n={r['n']:4d}  CD off: median {r['cd_off_median']:.2f}%  95th {r['cd_off_p95']:.2f}%  worst {r['cd_off_max']:.2f}%  (alpha 15: {r['cd_off_a15_p95']:.2f}%)   CD on: 95th {r['cd_on_p95']:.2f}%"
                if have_nf:
                    cn = 100 * np.array(e["cn"]); cp = np.array(e["cp"]); cpb = np.array(e["cp_bt"])
                    r.update({"cn_p95": float(np.percentile(cn, 95)), "cn_max": float(cn.max()), "cp_p95_in": float(np.percentile(cp, 95)), "cp_max_in": float(cp.max()),
                              "cp_boattail_max_in": float(cpb.max()), "distribution_closure": float(max(e["dist"])) if e["dist"] else None})
                    line += f"   CN: 95th {r['cn_p95']:.2f}% worst {r['cn_max']:.2f}%   CP: 95th {r['cp_p95_in']:.2f} in worst {r['cp_max_in']:.2f} in"
                if have_split:
                    sp = np.array(e["split"])
                    r.update({"ca_split_median_pts": float(np.median(sp)), "ca_split_worst_pts": float(sp.max()),
                              "ca_distribution_closure": float(max(e["ca_dist"])) if e["ca_dist"] else None})
                    line += f"   CA split: median {r['ca_split_median_pts']:.2f} pts worst {r['ca_split_worst_pts']:.2f} pts"
                out[key] = r; log(line)
        return out


if __name__ == "__main__":
    m = DragModel()
    print(f"dragmodel.h5: {m.info.get('solver', '')}, built {m.info.get('created', '')}")
    print("worst-case error over the Mach grid, per vehicle, against RASAero II runs the model never saw (CN at alpha 5 to 15; CP in inches over all alpha;")
    print("CA split: worst error in any component's share of CD, in percentage points, outside the transonic fairing):")
    m.verify()
