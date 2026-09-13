import numpy as np
from Vehicle.Vehicle import Vehicle
from Vehicle.Engine import Engine
from Configs.loader import load_config
from FluidProperties.PropertyModels import CoolPropPropertySource
from plot import plot_vehicle, plot_rocket_3d


def main():

    # -------------------------------------------------
    # Load configuration
    # -------------------------------------------------
    cfg = load_config("Configs/flight_2500lbf_mr2.yaml")

    # -------------------------------------------------
    # Create engine and configured vehicle
    # -------------------------------------------------
    engine = Engine(
        mass=cfg["engine"]["mass"],
        length=cfg["engine"]["length"],
        exit_area=0.025,
    )

    # -------------------------------------------------
    # Build Vehicle
    # -------------------------------------------------
    vehicle = Vehicle(cfg, CoolPropPropertySource())
    vehicle.build(engine)
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
