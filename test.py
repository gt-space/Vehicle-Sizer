import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from Vehicle.Vehicle import Vehicle
from Vehicle.InterTank import InterTank
from Vehicle.PressTank import PressTank
from Vehicle.FinCan import FinCan
#from Vehicle.PropTank import PropTank
from Vehicle.Nosecone import Nosecone
from Vehicle.COPV import COPV
from Configs.loader import load_config


def main():

    # -------------------------------------------------
    # Load configuration
    # -------------------------------------------------
    cfg = load_config("Configs/kerolox_pumped.yaml")

    # -------------------------------------------------
    # Create Sections
    # -------------------------------------------------

    s8 = Nosecone(cfg)

    # Intertank 1
    L1 = 0.12
    P1 = 6000.0
    M1 = 5000.0
    s1 = InterTank(cfg, L1, P1, M1)

    # Press Tank + COPV
    copv1 = COPV(
        volume=0.040,
        mass=18.2,
        length=1.4,
        diameter=0.25
    )

    copv2 = COPV(
        volume=0.020,
        mass=7.0,
        length=0.661,
        diameter=0.25
    )
    s2 = PressTank(cfg, copv2)

    # Intertank 2
    L3 = 0.15
    P3 = 12000.0
    M3 = 10000.0
    s3 = InterTank(cfg, L3, P3, M3)

    s4 = FinCan(cfg)

    #s5 = PropTank(cfg, medium="oxygen", prop_mass=200, material="aluminum_6061_t6", passthrough_diameter=0.052, ellipse_ratio=1.5, ullage_factor=1.1)
    #s6 = PropTank(cfg, medium="n-Dodecane", prop_mass=100, material="aluminum_6061_t6", passthrough_diameter=0.05, ellipse_ratio=1.5, ullage_factor=1.1)

    L7 = 0.36
    P7 = 15000.0
    M7 = 7000.0
    s7 = InterTank(cfg, L7, P7, M7)

    sections = [s8, s2, s1, s3, s7, s4]

    # -------------------------------------------------
    # Build Vehicle
    # -------------------------------------------------
    vehicle = Vehicle(cfg, sections)
    vehicle.build()
    vehicle.get_CNa(M=3, alpha=0.1)
    q=100e3
    A=1.0
    alpha_vec=np.full(vehicle.n, 0.1)
    N=vehicle.CNa*alpha_vec*q*A

    SM = (vehicle.cp - vehicle.cg) / cfg["vehicle"]["OMLD"]

    # -------------------------------------------------
    # Print Results
    # -------------------------------------------------
    print("\n--- SECTION DATA ---")
    for i, sec in enumerate(vehicle.sections):
        print(f"\nSection {i+1}")
        print("  Length:", sec.length)
        print("  Mass:", np.sum(sec.mass))
        print("  EI:", sec.EI[0])

    print("\n--- VEHICLE DATA ---")
    print("Total Length:", vehicle.length)
    print("Total Mass:", vehicle.total_mass)
    print("CG:", vehicle.cg)
    print("CP:", vehicle.cp)
    print("SM:", SM)

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
    plt.title("Mass and CNa Distribution")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()