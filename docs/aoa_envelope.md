# Roll–pitch–yaw AoA envelope

`Vehicle.utils.aoa.aoa_from_flight_history` consumes the dictionaries returned by
`FlightSim.run()`. It uses their time, speed, dynamic pressure, roll inertia,
and pitch inertia, plus `RocketAeroGeometry` reference area, diameter and cant.
It is a postprocessor; it does not change the flight solver's scheduled AoA.
Initial amplitudes and roll rate refer to the first supplied snapshot (the
existing solver returns endpoint snapshots, not the launch state).

```python
from Vehicle.utils.aoa import Cl_delta, aoa_from_flight_history

# geometry is a populated RocketAeroGeometry; flight_history = flight.run().
# Supply these quantities from a coefficient model or measured assumptions:
# single_fin_CNa, radius_at_fin_LE, radius_at_fin_TE, Cm_alpha, Cm_q, Cl_p,
# yaw_inertia, initial_roll_rate, initial_mode_1_rad, initial_mode_2_rad.
roll_derivative = Cl_delta(
    geometry.fin_count, geometry.semi_span,
    radius_at_fin_LE, radius_at_fin_TE,
    single_fin_CNa, geometry.diameter,
)
result = aoa_from_flight_history(
    flight_history, geometry, Iz=yaw_inertia,
    Cm_alpha=Cm_alpha, Cm_q=Cm_q,
    Cl_delta=roll_derivative, Cl_p=Cl_p,
    p0=initial_roll_rate,
    R1_0=initial_mode_1_rad, R2_0=initial_mode_2_rad,
)
aoa_rad = result['alpha_upper']
peak_aoa_deg = result['peak_alpha_deg']
peak_q_aoa = result['peak_q_alpha']  # Pa*rad
```

These required keyword inputs are the integration hooks for values absent from
the existing interfaces; no invented derivative or yaw-inertia defaults are
used. Coefficients and inertias accept scalars or histories matching the flight
samples. Inputs are used without validation or cleanup; callers supply consistent
shapes and physical units. Evaluate Mach-dependent derivatives upstream. `Cl_delta` requires the
single-fin normal-force derivative, not the existing whole-fin-set derivative.
The adapter converts geometry cant from degrees to radians. The lower-level
`roll_pitch_yaw_upper_envelope` accepts arrays directly and uses radians.

Supply `roll_rate=p_history` if another solver already integrates roll; this
bypasses internal roll integration. Otherwise roll uses explicit Euler.
Optional `M_pitch_external`, `M_yaw_external`, and `M_roll_external` accept
scalars or histories in N*m and default to zero.

Returned fields include `p`, `R1`, `R2`, `lambda_1`, `lambda_2`, `omega_1`,
`omega_2`, `alpha_trim`, `alpha_upper`, `q_alpha_upper`, peaks and peak times.
`diagnostic_modes` flags samples without two oscillatory conjugate pairs.
Modal propagation uses the preceding sample's growth rate, following the
handoff equation rather than its inconsistent sample-code indexing. The
non-oscillatory fallback retains one representative per conjugate pair and
selects the largest growth rates so real unstable roots are not hidden.

This is the handoff's simplified conservative envelope, not a validated,
phase-resolved AoA predictor. It assumes principal body axes, omits translational
normal-force and Cm_dot_alpha coupling, and does not model excitation from
changing trim/inertia. Frequency sorting can swap mode identities at crossings;
the non-oscillatory fallback is diagnostic. At vanishing stiffness, static trim
is set to zero and external transverse forcing is not propagated dynamically.
Absolute amplitudes depend on the supplied initial modal amplitudes. Compare
refined time grids before interpreting peaks.
