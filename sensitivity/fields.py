"""Classify saved sweep fields for discovery."""


def field_category(name: str) -> str:
    """Group available fields for discovery; not a fixed output whitelist."""
    low = name.lower()
    if 'static_stability' in low or 'static_margin' in low or 'stability_caliber' in low:
        return 'Stability'
    if low.startswith(('constraints.', 'constraint_', 'geometry_constraints.')) or low in ('feasible', 'accepted'):
        return 'Constraints'
    if 'thermal' in low or 'heat_' in low or 'temperature' in low or '.t_liq' in low or '.t_ull' in low:
        if any(token in low for token in ('press_tank', 'pressurant', 'copv')):
            return 'COPV'
        return 'Thermal'
    if any(token in low for token in ('press_tank', 'pressurant', 'copv')):
        return 'COPV'
    if any(token in low for token in ('ox_tank', 'fuel_tank', 'tank.')):
        return 'Propellant tanks'
    if any(token in low for token in ('pump', 'battery', 'motor_', 'electrical')):
        return 'Electrical / pumps'
    if any(token in low for token in ('gg_', 'gas_generator', 'gasgenerator')):
        return 'Gas generator'
    if any(token in low for token in ('regulator', 'valve', 'pressure_tracking', 'transition', 'switch_count')):
        return 'Controls'
    if any(token in low for token in ('loads.', 'bending', 'shear', 'axial_load', 'load_peaks', 'stress')):
        return 'Structural'
    if any(token in low for token in ('aero.', 'state.atmosphere.', 'kinematics.alpha', 'mach', 'dynamic_pressure', 'drag_', 'cd', 'angle_of_attack', 'aoa')):
        return 'Aerodynamics'
    if any(token in low for token in ('cg_', '.cg', 'inertia', 'iyy', 'ixx')):
        return 'Mass properties'
    if low.startswith(('design_summary.', 'vehicle.')) or any(token in low for token in ('dry_mass', 'initial_mass', 'fineness_ratio')):
        return 'Vehicle sizing'
    if any(token in low for token in ('thrust', 'chamber_pressure', 'mixture_ratio', 'mdot', 'propulsion.', 'impulse')):
        return 'Propulsion'
    if any(token in low for token in ('flight.', 'kinematics.', 'state.wind.', 'derived.speed_', 'apogee', 'velocity', 'altitude', 'final_x', 'final_v', 'burn_duration')):
        return 'Flight performance'
    return 'Other'
