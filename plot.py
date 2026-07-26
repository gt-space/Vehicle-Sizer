import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Vehicle.Vehicle import Vehicle
from Vehicle.sections.Nosecone import Nosecone
from Vehicle.sections.AviBay import AviBay
from Vehicle.sections.PressTank import PressTank
from Vehicle.sections.PropTank import PropTank
from Vehicle.sections.InterTank import InterTank
from Vehicle.sections.FinCan import FinCan


def plot_vehicle(vehicle: Vehicle, OMLD: float):

    SM = (vehicle.cp - vehicle.cg) / OMLD

    fig, ax1 = plt.subplots()

    ax1.plot(vehicle.station, vehicle.mass, label="Mass", color="black")
    ax1.axvline(vehicle.cg, color="red", linestyle="--", linewidth=2, label="CG")
    ax1.set_xlabel("Station (m)")
    ax1.set_ylabel("Mass per slice (kg)")

    ax2 = ax1.twinx()
    ax2.plot(vehicle.station, vehicle.CNa, linestyle=":", label="CNa", color="blue")
    ax2.axvline(vehicle.cp, color="green", linestyle="--", linewidth=2, label="CP")
    ax2.set_ylabel("CNa")

    fig.legend(loc="upper right")
    plt.title(f"Mass and CNa Distribution  |  SM = {SM:.3f} cal")
    plt.grid(True)
    plt.show()


def plot_rocket_3d(vehicle: Vehicle, cfg: dict):

    N_theta = 60
    theta = np.linspace(0, 2 * np.pi, N_theta)

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111, projection="3d")

    for sec in vehicle.sections:
        x, r = _section_profile(sec, cfg)
        color = _section_color(sec)

        X = np.outer(x, np.ones(N_theta))
        Y = np.outer(r, np.cos(theta))
        Z = np.outer(r, np.sin(theta))

        ax.plot_surface(X, Y, Z, color=color, alpha=0.9, linewidth=0, antialiased=True)

        if isinstance(sec, FinCan):
            _add_fins(ax, sec, cfg)

    L = vehicle.length
    R = float(cfg["vehicle"]["OMLD"]) * 0.5
    fin_span = 0.0
    for sec in vehicle.sections:
        if isinstance(sec, FinCan):
            Cr = sec.length
            Ct = Cr / 3.0
            fin_span = 2.0 * float(cfg["fin_can"]["fin_area"]) / (Cr + Ct)
            break
    R_max = R + fin_span

    ax.set_proj_type("ortho")
    ax.set_xlim(0, L)
    ax.set_ylim(-R_max, R_max)
    ax.set_zlim(-R_max, R_max)
    ax.set_box_aspect([L, 2 * R_max, 2 * R_max])
    ax.set_xlabel("Station (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Rocket Geometry (to scale)")
    plt.tight_layout()
    plt.show()


def _section_profile(sec, cfg):
    if isinstance(sec, Nosecone):
        x = np.concatenate([[sec.start_station], sec.station])
        r = np.concatenate([[0.0], sec.radius])
        return x, r

    if hasattr(sec, "radius") and isinstance(sec.radius, np.ndarray):
        return sec.station, sec.radius

    if isinstance(sec, FinCan):
        Ae = 0.025
        r_s = cfg["vehicle"]["OMLD"] * 0.5
        t = cfg["fin_can"]["boattail_wall_thickness"]
        r_f = np.sqrt(Ae / np.pi) + t
        x_local = np.arange(sec.n) * sec.dx
        r = r_s + (x_local / sec.length) * (r_f - r_s)
        return sec.station, r

    # cylindrical — only need end points
    R = cfg["vehicle"]["OMLD"] * 0.5
    x = np.array([sec.start_station, sec.end_station])
    r = np.array([R, R])
    return x, r


def _section_color(sec):
    if isinstance(sec, Nosecone):  return "#cccccc"
    if isinstance(sec, AviBay):    return "#aaddff"
    if isinstance(sec, PressTank): return "#ffbb55"
    if isinstance(sec, PropTank):
        return "#55aaff" if "oxygen" in sec.medium.lower() else "#ffee66"
    if isinstance(sec, InterTank): return "#999999"
    if isinstance(sec, FinCan):    return "#666666"
    return "#bbbbbb"


def _add_fins(ax, sec, cfg):
    N = cfg["fin_can"]["fin_count"]
    Cr = sec.length
    Ct = Cr / 3
    fin_area = cfg["fin_can"]["fin_area"]
    span = 2 * fin_area / (Cr + Ct)

    r_s = cfg["vehicle"]["OMLD"] * 0.5
    t = cfg["fin_can"]["boattail_wall_thickness"]
    r_f = np.sqrt(0.025 / np.pi) + t  # boattail exit radius (matches FinCan hardcoded Ae)
    x0 = sec.start_station
    sweep = (Cr - Ct) / 2

    r_tip = r_s + span

    for k in range(N):
        phi = k * 2 * np.pi / N
        cp, sp = np.cos(phi), np.sin(phi)

        # Vertical fin panel at constant body radius
        fin_verts = [[
            [x0,               r_s   * cp, r_s   * sp],
            [x0 + Cr,          r_s   * cp, r_s   * sp],
            [x0 + sweep + Ct,  r_tip * cp, r_tip * sp],
            [x0 + sweep,       r_tip * cp, r_tip * sp],
        ]]
        poly = Poly3DCollection(fin_verts, alpha=0.85, facecolor="#444444", edgecolor="#222222", linewidth=0.5)
        ax.add_collection3d(poly)

        # Triangular filler between fin root and boattail taper
        filler_verts = [[
            [x0,      r_s * cp, r_s * sp],
            [x0 + Cr, r_s * cp, r_s * sp],
            [x0 + Cr, r_f * cp, r_f * sp],
        ]]
        filler = Poly3DCollection(filler_verts, alpha=0.85, facecolor="#444444", edgecolor="#222222", linewidth=0.5)
        ax.add_collection3d(filler)
