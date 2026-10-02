# Canard stability and center of pressure

The base vehicle's normal-force coefficient and CP come from the [RASAero-derived aero model](../AeroTables/README.md), evaluated at the current Mach and absolute AoA. That CP, `aero.cp`, excludes canards. [`flight_forces.py`](flight_forces.py) leaves the base aero-table interpolation unchanged. When canards are present, [`Flight.py`](Flight.py) calculates `aero.total_cp` for stability reporting. Pitch dynamics continue to use the base aero-table CP and the canard's separate pitch moment.

The reported margin in calibers is `(total_cp - cg) / diameter`, with axial stations measured aft from the nose. A positive margin puts CP behind CG. Pitch dynamics use:

```text
M_vehicle = -N_vehicle * (cp_vehicle - cg)
M_canard  = -N_canard * (station_canard - cg)
M_pitch   = M_vehicle + M_canard + M_tilted_thrust
```

The canard's `pitch_moment_nm` supplies `M_canard`. The slope-based `total_cp` does not enter this equation.

## Two CP methods

The normal method places the instantaneous combined normal force at its force-weighted CP:

```text
x_force = (N_vehicle * x_vehicle + N_canard * x_canard)
          / (N_vehicle + N_canard)
```

If the resulting margin is below the configured requirement, or the total normal force is too close to zero to define that CP, the code uses the AoA-slope estimate:

```text
CNa_total = CNa_vehicle + CNa_canard
x_slope   = (CNa_vehicle * cg - Cma_vehicle * diameter
             + CNa_canard * x_canard) / CNa_total
```

The two methods weight different quantities: `x_force` weights the normal forces **at the current AoA**, while `x_slope` weights how those forces **change with AoA**. The base vehicle's location in the slope calculation is also derived from that change:

$$
x_{\mathrm{vehicle,slope}} =
\frac{C_{N,1}x_{\mathrm{vehicle},1}-C_{N,0}x_{\mathrm{vehicle},0}}
     {C_{N,1}-C_{N,0}}.
$$

Here, points 0 and 1 are the two aero-table AoA points bracketing the current magnitude of AoA, after interpolation at the current Mach. Thus a small current vehicle normal force can let the canard dominate `x_force`, even when the vehicle has a substantial AoA slope and carries substantial weight in `x_slope`. A small force does not imply a small slope. The canard derivative uses a small central difference of the undeflected canard lift model. Differential roll deflection does **not** directly change canard pitch force, its AoA slope, or CP; it still changes roll control and drag.

The switch is checked on every aerodynamic evaluation and is not latched. If `constraints.min_stability_calibers` is present, it is the threshold. Otherwise `canards.min_stability_calibers` is used. A normal `main.py` run removes the `constraints` section unless `--enforce-constraints` is passed, so the [pressure-fed regulator config](../Configs/flight_pressure_fed_regulator.yaml) also sets `canards.min_stability_calibers: 2.0`.

Set `canards.canard_stablity_on: false` to remove canard normal force, pitch moment, and CP shift while keeping canard roll control and drag. The spelling `stablity` matches the current configuration key. With no canards, or with this setting `false`, `total_cp` remains the base vehicle CP.

## Interpretation

The slope-based CP changes stability reporting when selected; it does not change the base aero interpolation, canard force magnitudes, or pitch dynamics. A reported margin above the threshold after fallback does not by itself prove that the underlying force model has become physically stable.

Near zero AoA at high Mach, PCHIP interpolation of the base table's total `Cn` can yield a very small vehicle normal force even though neighboring table points have a substantial secant slope. The force-weighted CP can then move almost to the canard station. The fallback uses table slopes to avoid that force ratio in the selected CP; it does not alter the underlying base `Cn`. The roll plot's pitch natural frequency also remains vehicle-only. The flight CSV records `cp_m` and `static_stability_margin_calibers` from `total_cp`, while its `Cn` and `normal_force_N` columns retain the base vehicle values and the canard normal force has its own column.
