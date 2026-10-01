# Optimizer constraints and preferences

## Nozzle design variable

`bounds.exit_pressure` varies `engine.exit_pressure` in Pa absolute. Both optimizer
configs use [40726, 103421] Pa (approximately 5.91-15.00 psia), rounded inward
from the combustion table's common sizing interval [40725.7712, 103421.3592] Pa
for chamber pressures 1.4-3.5 MPa, mixture ratios 1.5-3.0, and nfz=1.

The table supports expansion ratios 1-10. Sizing queries its ambient axis at the
design exit pressure, so that axis also bounds the interval. Startup validation
checks the configured Pc/MR rectangle at its boundaries and all enclosed table
knots, including ambient knots, and rejects exit-pressure bounds outside the
common inversion bracket. Widening Pc/MR bounds may require new exit bounds.

This is a fixed nozzle design target, not a prescribed exit pressure during
flight. Existing sizing matches target thrust at ambient pressure equal to design
exit pressure, then derives throat area, exit area, and design mass flows.
Geometry constraints still apply; table support does not guarantee a nozzle fits.
At popsize=4 there are now 68 pressure-fed or 76 pump-fed population members.
Old coordinate vectors are incompatible without explicit variable migration;
start a new output directory. Historical configs and result scores are unchanged.

The regulated flight-level hard limits are owned by `evaluation_overrides` in
the optimizer YAMLs. The pressure-fed search also sets maximum total vehicle
length/body diameter to 25 there. Base regulated flight configs no longer define
these search-specific limits. Tank temperature/pressure floors remain owned by
the tanks in the flight configs, including the COPV temperature used in sizing.
`hard_constraints` in the optimizer YAML declares required effective limits and
is validated after overrides are applied to an in-memory copy of the flight config.
Saved base/candidate/verified YAMLs retain the applied limits for standalone replay.
The intrinsic 15-degree aerodynamic flight guard remains even without an override.
It is not an opt-out list: physical, geometry, and other configured flight limits
remain mandatory even if omitted from this declaration.

The four flight configurations use these hard tank limits (absolute pressure):

| Owner | Minimum temperature | Minimum pressure |
| --- | --- | --- |
| Fuel | 80 K, liquid and ullage | 10 psia |
| Oxidizer | 80 K, liquid and ullage | 10 psia |
| COPV | 220 K | 600 psia |

The minimum stability margin is 2 calibers. Existing mission limits remain hard,
including the 75 s burn-duration ceiling, 150 km apogee goal, and 15 degree AoA
ceiling. Tank floor margins are retained across accepted fluid states; flight
limits use the existing trajectory constraint tracking. Violations make a design
infeasible. Missing required assessments cannot establish feasibility.

Choking margins no longer contribute constraints or tank floors. Legacy
`require_choked` branch settings are rejected with a migration message. Physical
choked-flow equations and the critical-pressure branch-sizing reference remain;
neither imposes an end-of-burn feasibility requirement.

## Soft penalties

`Configs/optimizer_pressure_fed.yaml` owns soft thresholds, weights and scales.
Fuel and oxidizer are tracked independently against each template node's operating
design pressure `P0`, not the vessel structural rating or a changing COPV pressure.

For each propellant tank:

```
I = integral during burn of abs(P(t) - P0) dt
penalty = 0.05 * I / (100000 Pa * 10 s)
```

The integral is two-sided and uses linear interpolation between accepted network
samples, including a zero-crossing correction. It stops at shutdown. Preview and
rollback checkpoints include the accumulators; rejected trials do not contribute.
The fixed 10 s scale is not the actual burn duration: longer deviations cost more.

Maximum dynamic pressure contributes:

```
penalty = 0.10 * max(0, max_q - 140000 Pa) / 10000 Pa
```

This is a peak penalty, not an exposure integral. Above 140 kPa remains feasible
when every hard constraint passes. A simultaneous hard max-Q limit is rejected
by this optimizer policy to avoid silently contradicting that choice.

## Ranking and reporting

Both optimizer YAMLs also configure a stability length-fraction soft band. Flight
tracks minimum and maximum `(CP - CG) / vehicle_length` on the same samples as
the hard caliber constraint (including coast). The added objective term is:

```
0.15 * (max(0, 0.10 - minimum_fraction) + max(0, maximum_fraction - 0.20)) / 0.05
```

The boundaries carry zero penalty. Both excursions add if a flight crosses both
limits. At a 250 kg mass reference, one percentage point outside either bound
costs 7.5 kg equivalent mass. This is not a direct L/D constraint; the hard
2-caliber minimum remains unchanged. Missing/nonfinite extrema cannot silently
receive zero penalty. Evaluation and verification JSON include both extrema.
Historical search scores are unchanged; use a new search directory for this policy.

For hard-feasible candidates, `J = initial_mass / 250 kg + sum(soft penalties)`
and `score = J / (1 + J)`. Lower is better. For example, 10 kPa above the max-Q
target adds 0.10 to J, equivalent to 25 kg at the configured mass reference.
Incumbents and search summaries select by score, not mass alone.

Completed hard-infeasible candidates rank in [1, 2), preflight rejections in
[2, 3), and unresolved failures at 3. Hard-violation scales guide ranking within
these categories; soft weights never turn a failed hard limit into feasibility.

Evaluation JSONL and verification JSON report the pressure integrals, durations,
targets, peak absolute errors, and soft contributions. Missing or invalid powered
tracking on an otherwise accepted candidate is an error, not a zero penalty.
No new native solver recovery or numerical failure suppression is introduced.
