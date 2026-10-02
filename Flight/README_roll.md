# Roll analysis

[`roll.py`](roll.py) advances roll rate and angle alongside the flight loop. It reads the current atmosphere, Mach, vehicle mass properties, thrust, and aerodynamic derivatives. Roll is independent of the planar position and pitch kinematics: it does not rotate the vehicle's force axes or add roll-induced AoA to the flight state.

The analysis starts at zero roll rate and zero roll angle. At each accepted flight sample, it advances the previous roll state using the previous net roll moment and the previous vehicle `Ixx`:

```text
p_next   = p + M_roll * dt / Ixx
phi_next = phi + p * dt
```

`Ixx` comes from the assembled vehicle and changes with its mass distribution. The roll analysis is constructed once per simulation run, then sampled inside that run's loop.

## Moments and aerodynamic inputs

The moment calculation separates forcing and damping:

```text
M_forcing = M_thrust + qbar * S * D * fin_cant * Cl_delta + M_external
M_damping = qbar * S * D * Cl_p * p * D / (2 * V)
M_open    = M_forcing + M_damping
M_net     = M_open + M_canard
```

Here `S = pi * D² / 4`, `p` is roll rate in rad/s, and `V` is speed. `Cl_p` is normally negative, so damping opposes roll. The finite-fin roll damping shape comes from [`Vehicle/utils/geometry.py`](../Vehicle/utils/geometry.py). See the [RocketPy roll-equation discussion](https://docs.rocketpy.org/en/latest/technical/aerodynamics/roll_equations.html) for the roll-damping formulation.

The `roll` config uses `fin_cant` in radians. `thrust_offset_y` is an engine offset in metres and `thrust_angle_z` is a misalignment angle in radians. Their roll torque is assigned the sign of the fin forcing so they combine constructively. `M_roll_external` is an optional moment in N m. Optional `Cl_delta`, `Cl_p`, and `Cm_alpha` values override calculated derivatives; otherwise the fin normal-force slope and pitch-moment slope come from the aero model at the current Mach. Roll starts at zero without `p0` or `phi0` settings.

When canards are configured, their controller tries to cancel `M_forcing`, not `M_open`. Damping therefore remains in the net moment and can slow a spinning vehicle. The controller's roll authority is multiplied by `canards.net_roll_multiplier`.

The damping denominator currently uses `hypot(kin.vx, kin.vz)`, while dynamic pressure and Mach come from the atmosphere calculation. With wind, that speed need not equal air-relative speed.

## History and plots

`main.py` writes a `_roll.csv` next to the flight CSV. It contains time, roll rate in rad/s, roll angle, thrust roll torque, forcing, damping, canard control, open and net moments, and pitch natural frequency. The configured plot path also produces `_roll_moments.png` (forcing, damping, canard control) and `_roll_frequencies.png` (roll rate converted to Hz and pitch natural frequency in Hz).

The plotted pitch natural frequency is calculated from the **vehicle-only** `Cm_alpha` and `Iyy`: `sqrt(-qbar * S * D * Cm_alpha / Iyy) / (2*pi)`. It is `NaN` when that stiffness is destabilizing. It does not include the canard CP switch described in [Stability](README_stability.md).
