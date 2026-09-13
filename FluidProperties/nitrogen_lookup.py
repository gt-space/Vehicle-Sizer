"""Generate pressurant-gas PT tables used by the runtime solver."""

try:
    from .pt_lookup import (
        PTSpecification,
        generate_pt_table,
        generate_saturation_table,
    )
except ImportError:
    from pt_lookup import PTSpecification, generate_pt_table, generate_saturation_table


SPECS = (
    PTSpecification(
        "Nitrogen", "nitrogen_pt", (1.0e3, 7.5e7), (63.151, 1200.0), 220, 600, "gas"
    ),
    PTSpecification(
        "Helium", "helium_pt", (1.0e3, 7.5e7), (10.0, 1200.0), 220, 300, "auto"
    ),
)


if __name__ == "__main__":
    for specification in SPECS:
        print(f"Generating /{specification.group}...")
        generate_pt_table(specification)
    print("Generating /nitrogen_saturation...")
    generate_saturation_table("Nitrogen", "nitrogen_saturation")
    print("Pressurant PT tables complete.")
