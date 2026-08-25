import numpy as np
from unittest.mock import MagicMock
from Vehicle.Vehicle import Vehicle
from Vehicle.sections import Nosecone, AviBay, InterTank, PressTank, FinCan, PropTank
from Vehicle.COPV import COPV
from Configs.loader import load_config
from plot import plot_vehicle, plot_rocket_3d


def main():

    # -------------------------------------------------
    # Load configuration
    # -------------------------------------------------
    cfg = load_config("Configs/kerolox_pumped.yaml")

    # -------------------------------------------------
    # Create Sections
    # -------------------------------------------------

    s8 = Nosecone(cfg)
    s9 = AviBay(cfg)

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
        diameter=0.25,
        internal_area=1.2,
    )

    copv2 = COPV(
        volume=0.020,
        mass=7.0,
        length=0.661,
        diameter=0.25,
        internal_area=0.7,
    )
    s2 = PressTank(cfg, copv2)

    # Intertank 2
    L3 = 0.15
    P3 = 12000.0
    M3 = 10000.0
    s3 = InterTank(cfg, L3, P3, M3)

    engine = MagicMock()   # Engine.py incomplete — stub until ready
    engine.mass = cfg["engine"]["mass"]
    engine.exit_area = 0.025  # m^2, consistent with hardcoded Ae in FinCan
    s4 = FinCan(cfg, engine)

    s5 = PropTank(cfg, medium="oxygen", prop_mass=200, material="aluminum_6061_t6", passthrough_diameter=0.052, ellipse_ratio=1.5, ullage_factor=1.1, P_liq0=1e6, T_liq0=90.0)
    s6 = PropTank(cfg, medium="n-Dodecane", prop_mass=100, material="aluminum_6061_t6", passthrough_diameter=0.05, ellipse_ratio=1.5, ullage_factor=1.1, P_liq0=1e6, T_liq0=300.0)

    L7 = 0.36
    P7 = 15000.0
    M7 = 7000.0
    s7 = InterTank(cfg, L7, P7, M7)

    sections = [s8, s9, s2, s1, s5, s3, s6, s7, s4]

    # -------------------------------------------------
    # Build Vehicle
    # -------------------------------------------------
    vehicle = Vehicle(cfg, engine, sections)
    vehicle.build()
    vehicle.get_CNa(M=3, alpha=0.1)
    q=100e3
    A=1.0
    alpha_vec=np.full(vehicle.n, 0.1)
    #N=vehicle.CNa*alpha_vec*q*A

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

    plot_vehicle(vehicle, cfg["vehicle"]["OMLD"])
    plot_rocket_3d(vehicle, cfg)

if __name__ == "__main__":
    main()
