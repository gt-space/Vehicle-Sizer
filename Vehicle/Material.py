from dataclasses import dataclass

from contextlib import redirect_stdout
from io import StringIO

# matproplib prints its registry on import; library simulation must stay quiet.
with redirect_stdout(StringIO()):
    import matproplib as mp


@dataclass(frozen=True)
class MaterialProperties:
    """Material values loaded once from matproplib for vehicle calculations."""

    name: str
    density: float
    yield_strength: float | None
    elastic_modulus: float | None
    specific_heat: float | None = None
    thermal_conductivity: float | None = None

    @classmethod
    def from_name(cls, name: str) -> "MaterialProperties":
        material = mp.db.get_material(name)
        if material is None:
            raise ValueError(f"Unknown material {name!r}")

        def optional(*names: str) -> float | None:
            for property_name in names:
                if property_name in material.properties:
                    return float(material.get(property_name))
            return None

        return cls(
            name=name,
            density=float(material.get("density")),
            yield_strength=optional("yield_strength", "yield_strength_0deg"),
            elastic_modulus=optional("elastic_modulus", "elastic_modulus_0deg"),
            specific_heat=optional("specific_heat"),
            thermal_conductivity=optional("thermal_conductivity"),
        )

    def require(self, name: str) -> float:
        value = getattr(self, name)
        if value is None:
            raise ValueError(f"Material {self.name!r} has no {name}")
        return value
