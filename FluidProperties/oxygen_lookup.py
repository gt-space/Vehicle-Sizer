"""Generate the non-vaporizing liquid-oxygen PT table."""

try:
    from .pt_lookup import PTSpecification, generate_pt_table
except ImportError:
    from pt_lookup import PTSpecification, generate_pt_table


SPEC = PTSpecification(
    "Oxygen", "oxygen_pt", (1.0e3, 7.0e6), (55.0, 138.0), 180, 240, "liquid"
)


if __name__ == "__main__":
    print(f"Generating /{SPEC.group}...")
    generate_pt_table(SPEC)
    print("Oxygen PT table complete.")
