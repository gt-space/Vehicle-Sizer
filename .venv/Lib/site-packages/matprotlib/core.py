
import numpy as np
from typing import Union, List, Dict, Optional, Any
import warnings

# --- 1. The Smart Property Wrapper ---
class Prop:
    """
    Represents a single physical property (e.g., Yield Strength).
    Can be a constant float or a temperature-dependent lookup table.
    """
    def __init__(self, name: str, data: Union[float, List[np.ndarray]], units: str = ""):
        self.name = name
        self.units = units
        self._data = data
        self._is_constant = isinstance(data, (float, int))
        
        # Validation
        if not self._is_constant:
            if not isinstance(data, list) or len(data) != 2:
                raise ValueError(f"Property '{name}' must be a float or [Temp_Array, Value_Array]")

    def get(self, T_kelvin: float) -> float:
        """Get value at specific temperature."""
        if self._is_constant:
            return float(self._data)
        
        temps, values = self._data[0], self._data[1]
        # np.interp handles the linear interpolation safely
        return float(np.interp(T_kelvin, temps, values))

    def __repr__(self):
        val_str = f"{self._data}" if self._is_constant else "Temp_Dependent_Array"
        return f"<{self.name}: {val_str} {self.units}>"

# --- 2. The Fatigue Handler ---
class FatigueProfile:
    """
    Handles S-N curves (Stress vs Cycles).
    Ideally, you have different curves for different temperatures.
    This implementation picks the curve closest to the requested Temp.
    """
    def __init__(self, curves: Dict[float, List[np.ndarray]]):
        # Format: { Temperature_K : [Cycles_Array, Stress_Array] }
        self.curves = curves

    def get_limit(self, cycles: float, T_kelvin: float) -> float:
        """Returns max stress for a given number of cycles at Temp T."""
        # 1. Find the closest temperature curve available
        avail_temps = np.array(list(self.curves.keys()))
        idx = (np.abs(avail_temps - T_kelvin)).argmin()
        closest_T = avail_temps[idx]
        
        if abs(closest_T - T_kelvin) > 50:
            warnings.warn(f"Fatigue data unavailable for {T_kelvin}K. Using curve for {closest_T}K.")

        # 2. Interpolate Stress vs Cycles for that curve
        cycle_arr, stress_arr = self.curves[closest_T]
        
        # Log-Log interpolation is usually standard for S-N, but Linear is used here for simplicity
        return float(np.interp(cycles, cycle_arr, stress_arr))

# --- 3. The Material Class ---
class Material:
    def __init__(self, name: str, category: str = "General", default_condition: str = "annealed"):
        self.name = name
        self.category = category
        self.default_condition = default_condition
        
        # Structure: { prop_name: { condition_name: Prop_Object } }
        self.properties: Dict[str, Dict[str, Prop]] = {}
        self.fatigue: Dict[str, FatigueProfile] = {}
        self.metadata: Dict[str, Any] = {}

    def add_prop(self, key: str, data, units: str = "", condition: str = None):
        """Add a standard table/value property for a specific condition."""
        cond = condition if condition else self.default_condition
        if key not in self.properties:
            self.properties[key] = {}
        self.properties[key][cond] = Prop(key, data, units)


    def add_custom_prop(self, prop_object: Prop, condition: str = None):
        cond = condition if condition else self.default_condition
        key = prop_object.name
        if key not in self.properties:
            self.properties[key] = {}
        self.properties[key][cond] = prop_object

    def add_fatigue(self, curve_data: Dict[float, List[np.ndarray]], condition: str = None):
        cond = condition if condition else self.default_condition
        self.fatigue[cond] = FatigueProfile(curve_data)

    def add_meta(self, key: str, value):
        self.metadata[key] = value
    
    def get_meta(self, key: str):
        """Fetch a specific piece of metadata."""
        if key not in self.metadata:
            raise KeyError(f"Metadata '{key}' not found in material '{self.name}'.")
        return self.metadata[key]

    def get(self, prop_name: str, T: float = 298.0, condition: str = None) -> float:
        target_cond = condition if condition else self.default_condition
        
        if prop_name not in self.properties:
            raise AttributeError(f"Material '{self.name}' has no property '{prop_name}'")
            
        # --- NEW: Fallback logic ---
        if target_cond not in self.properties[prop_name]:
            # If the specific condition (e.g. "Aged") is missing, try the material's original baseline
            if "Annealed" in self.properties[prop_name]: 
                 target_cond = "Annealed"
            elif "Standard" in self.properties[prop_name]:
                 target_cond = "Standard"
            else:
                 # Just grab whatever the first available condition is as a last resort
                 target_cond = list(self.properties[prop_name].keys())[0]

        return self.properties[prop_name][target_cond].get(T)

    def __getattr__(self, item: str):
        if item.startswith('__'):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{item}'")
        props = self.__dict__.get('properties', {})
        if item in props:
            return self.get(item, T=298.0) 
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __repr__(self):
        return f"Material(Name='{self.name}', Default='{self.default_condition}', Props={len(self.properties)})"


# --- 4. The Database Registry ---
class MaterialRegistry:
    def __init__(self):
        self._db: Dict[str, Material] = {}

    def add_material(self, material: Material):
        key = material.name.replace(" ", "_").lower()
        self._db[key] = material

    def get_material(self, name_key: str) -> Material:
        key = name_key.replace(" ", "_").lower()
        return self._db.get(key)

    def list_materials(self):
        return list(self._db.keys())

# --- 5. Module-Level API (The "mp" Interface) ---
_default_registry = MaterialRegistry()

def add_material(material: Material):
    """Module-level wrapper to add a material to the default registry."""
    _default_registry.add_material(material)

def get_material(name_key: str) -> Material:
    """Module-level wrapper to fetch a material."""
    return _default_registry.get_material(name_key)

def list_materials() -> List[str]:
    """Module-level wrapper to list available materials."""
    return _default_registry.list_materials()
