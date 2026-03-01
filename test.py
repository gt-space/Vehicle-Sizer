import numpy as np

from Vehicle.Vehicle import Vehicle
from Vehicle.InterTank import InterTank
from Vehicle.PressTank import PressTank
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

    # Intertank 1
    L1 = 0.12
    P1 = 6000.0
    M1 = 10000.0
    s1 = InterTank(cfg, L1, P1, M1)

    # Press Tank + COPV
    copv = COPV(
        volume=0.012,
        mass=18.2,
        length=1.4,
        diameter=0.25
    )

    s2 = PressTank(cfg, copv)

    # Intertank 2
    L3 = 0.36
    P3 = 15000.0
    M3 = 5000.0
    s3 = InterTank(cfg, L3, P3, M3)

    sections = [s1, s2, s3]

    # -------------------------------------------------
    # Build Vehicle
    # -------------------------------------------------
    vehicle = Vehicle(cfg, sections)
    vehicle.build()

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

if __name__ == "__main__":
    main()