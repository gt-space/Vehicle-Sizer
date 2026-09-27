"""Generate the n-dodecane surrogate liquid-fuel PT table."""

try:
    from .pt_lookup import PTSpecification, generate_pt_table
except ImportError:
    from pt_lookup import PTSpecification, generate_pt_table


SPEC = PTSpecification(
    "n-Dodecane",
    "ndodecane_pt",
    (1.0e3, 7.0e6),
    (264.0, 600.0),
    180,
    260,
    "liquid",
)


if __name__ == "__main__":
    print(f"Generating /{SPEC.group}...")
    generate_pt_table(SPEC)
    print("n-Dodecane PT table complete.")
