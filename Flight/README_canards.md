# Canards

The canards are two triangular surfaces modeled by [`canards.py`](canards.py). Add both `canards` and `roll` sections to the flight configuration. `FlightSim.trial_aero()` evaluates them during each aerodynamic flight trial; they are not a postprocessing step. The canard drag acts on the trajectory, the commanded lift supplies roll control, and the undeflected pair can supply pitch-plane normal force when stability is enabled.

## Geometry and inputs

The [pressure-fed regulator config](../Configs/flight_pressure_fed_regulator.yaml) shows the current inputs:

| Input | Meaning |
| --- | --- |
| `root_chord_m` | Root chord of each triangular canard. A 74° leading-edge sweep fixes span as `root_chord_m / tan(74°)`; tip chord is zero. |
| `max_deflection_deg` | Absolute command limit. The command is also limited by the available lift-model angle after accounting for vehicle AoA. |
| `control_start_time_s` | Earliest time the roll controller may command deflection. It remains inactive on the launch rail. |
| `net_roll_multiplier` | Multiplies available and commanded roll torque. It does not multiply drag or pitch-plane normal force. |
| `station_m` | Canard aerodynamic-center station from the nose. If `null`, the model uses the front of the vehicle section containing a `canards` mass entry plus two-thirds of the root chord. |
| `canard_stablity_on` | Exact key used by the code. `false` removes canard normal force, pitch moment, and CP shift; roll control and drag remain. Defaults to `true`. |
| `min_stability_calibers` | CP-method switch threshold when the flight config has no enforced stability constraint. See [Stability](README_stability.md). |
| `aero.cd0`, `aero.oswald` | Zero-lift drag coefficient and subsonic induced-drag efficiency. |
| `aero.deflection_drag_multiplier` | Multiplies only drag added by nonzero deflection, after neutral drag has been calculated. |

The surface area and aspect ratio are calculated from the root chord and span. The code also applies its existing body interference factor to canard lift. The automatic axial station is an approximation; set `station_m` explicitly if the mounted aerodynamic center is known.

## Lift, control, and drag

`CanardAerodynamics.canard_lift()` uses the stored Polhamus coefficients through Mach 2.8. Above that it blends into a shock-expansion normal-force model corrected for finite triangular-wing lift; the blend reaches the high-Mach model at Mach 4. The finite-wing method distinguishes leading-edge regimes using `sqrt(M² - 1) cot(74°)`. The implementation cites [NACA TR-1050](https://ntrs.nasa.gov/citations/19930092096) and [NACA TR-970](https://ntrs.nasa.gov/citations/19930091081). These are isolated-surface estimates, not a calculation of the canard wake on downstream fins.

`CanardSystem.command_deflection()` finds the differential deflection needed to oppose the **roll forcing moment** at the current timestep, up to the configured limit. It does not command cancellation of roll damping. The reported `saturated` flag says the requested forcing cancellation exceeded modeled control authority. Deflection continues to determine canard drag even though it no longer enters the pitch-stability calculation.

The drag model includes neutral drag and drag from the commanded deflection. Below Mach 0.8 it uses a finite-wing induced-drag term; above Mach 1.2 it uses the supersonic lift-dependent term, with a blend in between. It reports `drag_n`, `neutral_drag_n`, and their difference. This drag is added to flight forces while aerodynamic evaluation is active, including when deflection is zero.

## Outputs and limits

`main.py` writes canard deflection, drag, roll authority, roll moment, normal force, and pitch-moment columns to the flight CSV. It also writes canard coefficient plots when canards are configured. The plots compare drag-related coefficient contributions at the same simulated states; they are not a second trajectory run.

The model does not calculate canard wake effects on the body or aft fins, a Mach-dependent canard aerodynamic-center station, or canard normal loads in the structural-load distribution. The `net_roll_multiplier` is an input for roll authority, not a general correction for those omitted effects.
