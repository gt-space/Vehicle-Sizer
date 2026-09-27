# Vehicle-Sizer

`Fluids/` contains the SUNDIALS/IDAS solver used by both flight and optimization.
The previous implicit solver and separate experimental package have been removed.

## Run

Use Python 3.12 with `sundials4py` (validated with 7.9.0), NumPy, SciPy,
CoolProp, RocketCEA, h5py, PyYAML, pandas and matplotlib. The sibling
`engine-designer/engineDesigner_v5.0/matproplib` package must be on `PYTHONPATH`.
Keep the property tables in `FluidTables/sizer_lookups.h5` and drag model in
`AeroTables/dragmodel.h5`; table generation is documented in those directories.

```bash
python main.py Configs/flight_pump_fed_regulator.yaml
python examples/run_propulsion.py Configs/flight_pressure_fed_bang_bang.yaml --duration 1 --dt 1
python optimizer.py Configs/optimizer.yaml --output outputs/optimization
python -m pytest -q
```

The four flight configs are `flight_pump_fed_regulator.yaml`,
`flight_pump_fed_bang_bang.yaml`, `flight_pressure_fed_regulator.yaml` and
`flight_pressure_fed_bang_bang.yaml`, under `Configs/`. `main.py` defaults to
the pump-fed regulator. Both optimizer configs use these same flight configs;
`optimizer_pressure_fed.yaml` selects the pressure-fed regulator.

Solver settings and component contracts: [Fluids/README.md](Fluids/README.md).
Custom wiring: [Configs/templates/README.md](Configs/templates/README.md).
Generated output belongs under `outputs/`; old benchmark/sweep artifacts have
been removed. `examples/run_optimizer_search.py` retains supervised workers and
optional diagnostic patches, and `examples/search_timing.py` records profiling.

## Simulation API

```python
from Configs.loader import load_config
from simulation import simulate

cfg = load_config("Configs/flight_pressure_fed_regulator.yaml")
result = simulate(cfg)
print(result.dry_mass, result.initial_mass, result.apogee)
```

`simulate` does not print, plot, write files, retain history, or compute beam
loads by default. It rebuilds vehicle and fluid state for every call without
mutating `cfg`. For repeated candidates, load `property_sources(cfg)` and
`AeroTables.DragModel` once and pass `pure_properties`, `combustion_properties`,
and `aero_model`. Reuse sources only while their table/fluid selections remain
unchanged. `record_history=True` retains endpoints. Independently,
`compute_loads=True` evaluates load distributions and retains absolute peaks
in `load_peaks`; distributions are stored only when history is enabled.
No flight structural acceptance constraints are imposed.
`main.py` is the separate CLI/CSV/plotting layer.

`simulation.fluid_stop_at_triple_point: true` optionally freezes the entire
fluid network when any stored fluid with saturation data reaches the lower
saturation-domain boundary (the table approximation to its triple point).
Saturated volumes use pressure; single-phase fluids use temperature, so warm
gas below the triple-point pressure does not trigger the cutoff. The event is
recorded as `triple_point:<fluid>`. Inventories and fluid states then remain
fixed, all flows and thrust are zero, and the trajectory continues to coast.
This is an idealized cutoff, not a model of freezing or sublimation. The option
defaults to false; fluids without saturation
data retain their existing property-domain checks.

`SimResult` exposes `initial_mass` (wet launch mass), `dry_mass`, `final_mass`,
`max_altitude`, `apogee`, `apogee_time`, `apogee_reached`, `max_q`,
`min_stability_calibers`, `burn_duration`, `burn_complete`, and `termination`.
An unfinished ascent has `apogee=None`, not an apogee equal to its last altitude.
Apogee is interpolated across a velocity sign change; other extrema and burn
duration use flight endpoints. Choose a sufficiently small step for constraints.

Optional `constraints` config entries: `goal_apogee` (minimum, metres), `max_q`
(Pa), `min_stability_calibers`, `min_rail_twr`, and `max_burn_duration` (s).
Results are signed margins: **nonnegative means satisfied**. An unfinished
apogee or undecidable burn limit returns `None`. `feasible` is false for any
violation, `None` for unassessed checks, and true only when all checks pass.
An optimizer can minimize either launch or dry mass; that choice is intentionally
left to its objective. Exceptions from the solver are not successful results.

`completed` means apogee reached. `accepted` additionally requires burn completion,
configured `goal_apogee` and `max_burn_duration`, and every required constraint
passing (equality passes). Without mission limits, diagnostic runs can be feasible
but are not accepted mission designs. There is no minimum burn duration, and
exceeding maximum burn duration does not stop integration. Check the final margin.
TWR is thrust divided by instantaneous weight, stored in history's `forces.twr`.
`min_rail_twr` and its time include initialization and the interpolated first rail
exit; `rail_exit_time` and `max_twr` are available without history. With no rail
interval, a requested rail constraint is unassessed rather than silently passing.

Example mission limits (choose values for the mission):

```yaml
constraints:
  goal_apogee: 10000.0       # m, same altitude datum as launch.altitude
  max_burn_duration: 60.0   # s; equality passes, no early termination
  min_rail_twr: 1.2         # dimensionless
```

Root `constraints.py` owns shared reporting and failure policy;
`Vehicle/geometry_constraints.py` owns packaging calculations. `constraint_records`
identifies each check's source, phase, units, required flag and worst-sample time
where available. Fluid minima use accepted substeps, not trial solves. Optional
unconfigured packaging checks are informational, not acceptance requirements.

## Design and initial conditions

`prop_system.initial_conditions` contains fluid identities, required `T`,
required `gas_T` for ullage, and optional propellant-tank `P` overrides.
Propellant pressure defaults to the propulsion design ladder (chamber pressure,
injector stiffness and feed losses/pump head). COPV initial pressure is exactly
`tanks.<id>.design_pressure`, the declared COPV charge design point.
There is no separate pressurant initial-pressure input or override mechanism.
There are no temperature defaults: missing, nonpositive, or nonfinite
temperatures are errors. When thermal simulation is enabled, both
`thermal.initial_temperature` and `thermal.sink_temperature` are also required.
Property-table domain checks still apply.

Derived masses and energies live in `PropSystem.initial_states`, never in the
design config. Vehicle builds recompute them from current volume, propellant
mass and P/T. The selected template is the single source of design pressures
and network wiring.

`advanced.weld_efficiency` defaults to 1.0 and can be overridden per tank.
Values in (0, 1] multiply material allowable strength before pressure sizing;
they do not multiply the computed thickness. Explicitly supplied gauges remain
inputs, with their required-gauge margin exposed as a constraint.

## Intertank hardware masses

Set `mass` (kg, finite and positive) on an `inter_tank` entry under
`vehicle.sections` to override its complete calculated mass:

```yaml
    - type: inter_tank
      length: 0.40
      area_moment_of_inertia: 3.60e-5
      mass: 11.3  # total section mass, including clamshell, stringers and hardware
```

When `mass` is omitted, the model sums calculated clamshell and stringer masses
plus optional per-section `feed_system_mass` and `avi_mass` (kg, nonnegative,
default zero). An explicit `mass` replaces that entire sum; nothing is added to
it. Geometry still determines stiffness and the thermal shell distribution.
The root `inter_tank` block holds shared materials, wall thickness and stringer
count only; hardware masses there are no longer read.

## Geometry and construction constraints

Stations are SI metres from the nose, increasing aft. `station` denotes cell
centers, `cell_edges` denotes boundaries, and `dx` within a section is its exact
length divided by its cell count (the configured `vehicle.dx` is a maximum).
Sections tile the body without gaps; fluid and thermal cells share that extent.
Pitch inertia retains the point-mass approximation.

The engine starts at the fin-can start and is independently distributed over
its own length into `vehicle.engine_mass` and `vehicle.dry_mass`. Fin-can mass
excludes it. An engine extending beyond the fin can is not modeled by the
current OML/aero geometry: construction raises `GeometryError`, carrying the
signed length constraint, instead of truncating or stretching engine mass.

For axial structural loading, the engine's forward end (`engine_start_station`)
is the assumed engine-to-airframe thrust interface. Thrust is lumped into the
cell immediately aft of that boundary, not the final vehicle cell/nozzle exit.
The distributed engine mass and inertial loads remain unchanged.

Construction returns COPV/airframe, tank/passthrough, wall-gauge, engine-length,
nozzle and optional engine-envelope/feedline margins. `simulate` returns known
negative margins in an infeasible result's `geometry_constraints`. Invalid
primitive dimensions can raise before an evaluable geometry exists; the API
wraps those exceptions as fatal `EvaluationFailure` instances.

Optional packaging inputs:

```yaml
engine:
  envelope_diameter: 0.20  # conservative cylindrical engine envelope, m
geometry:
  radial_clearance: 0.002  # minimum radial clearance, m
  feedlines:
    - name: oxidizer
      through_tank: fuel_tank
      outer_diameter: 0.0254
```

Feedline checks assess each specified line against a passthrough bore, not
multi-line bundle packing, bends, fittings or full routing CAD. Missing engine
envelope/feedline inputs are unassessed (`None`), not silently feasible.
These packaging checks do not certify COPV pressure ratings or weld strength.

## Pressurant design variables and feasibility

Each pressurant tank accepts independent design inputs; neither pressure nor
volume is solved from a closed-form blowdown equation:

```yaml
tanks:
  press_tank:
    type: pressurant
    volume: 0.100                 # internal gas volume, m^3
    design_pressure: 29647456.36  # Pa absolute; design AND initial charge pressure
    min_temperature: 220.0       # K; required for every pressurant tank
    material: 1018_carbon_steel_standard
    # mass: 18.2                  # kg; optional override of calculated shell mass
    # min_pressure: 6000000.0     # optional additional absolute pressure floor
    # ... existing diameter, wall, cap shape and mass inputs ...
prop_system:
  initial_conditions:
    press_tank:
      fluid: Nitrogen
      T: 300.0                   # required; no fallback
```

An optimizer varies `tanks.<id>.volume` and `design_pressure`; these are also
the physical initial volume and pressure, with no separate override. Legacy
`volume_liters` remains supported, but cannot be specified alongside `volume`.
Each build recomputes gas density, mass and energy from volume and initial P/T.
When dry mass is omitted, COPV shell mass is calculated from material density
and the configured cylindrical-wall and ellipsoidal-endcap volumes. An explicit
`mass` overrides that estimate.
Middle-of-life orifice sizing is retained, including the existing duty-cycle or
capacity-factor multiplier. Tank-level Tmin is its sizing temperature reference;
branch-level `min_temperature` is accepted when tank-level Tmin is absent.

`simulate(...).constraints` includes signed margins (nonnegative passes):

| Key | Margin | Units |
| --- | --- | --- |
| `tank.<id>.Tmin` | smallest checked tank temperature minus its limit | K |
| `branch.<id>.choked` | smallest checked upstream pressure minus downstream pressure divided by critical pressure ratio | Pa |
| `tank.<id>.Pmin` | worst of its outgoing choking margins and optional fixed pressure-floor margin | Pa |

The critical ratio is `(2/(gamma+1))**(gamma/(gamma-1))`, using the current
upstream gas gamma and bulk pressures, consistently with the current flow model.
Regulator, bang-bang and compressible-loss COPV feeds require choking. Other branches
can explicitly request `require_choked: true`. Closed pressurant valves are
checked for *available* choking capability, not treated as actual flowing jets.
Shared supplies must satisfy every outgoing branch's requirement.

Checks cover initialization and accepted fluid substep endpoints through engine
shutdown, not subsequent coast. Minimum margins accumulate even with history
disabled; trial solves do not latch failures into committed results. Endpoint
sampling is not a proof of continuous-time extrema: use timestep convergence
checks. Negative margins make `SimResult.feasible` false. Initial physical-limit
failures return `termination="infeasible_initial_design"` with signed margins
without running a trajectory. Ordinary packaging rejection follows the same
policy. Unavailable masses and requested flight margins are `None`, not infinity
or fabricated trajectory results: an optimizer adapter must support partial
evaluations. A converged liquid-pump solution requiring negative absolute inlet
pressure returns `termination="infeasible_operating_state"`, with a signed
`node.<inlet>.Pmin` margin in Pa and the attempted state time. The invalid state
is not committed and no apogee is fabricated. Signed inlet-pressure trials are
allowed only for the liquid pump equations; gas and property-table bounds are
unchanged. This check does not model cavitation or establish an NPSH margin.
Invalid configs, property-domain failures, other nonphysical calculations
and solver failures raise `EvaluationFailure`, with phase, copied config, partial
flight result where available, and the original exception chained. Let this stop
the optimizer for debugging; do not convert it to an infeasibility penalty.

COPV wall thickness is **still a supplied input**. Dry mass is estimated from
that wall geometry and material unless explicitly overridden. Volume changes
geometry and gas inventory; pressure changes gas inventory and orifice sizing,
but neither currently sizes a qualified pressure-vessel structure. Thus this is
not yet a physically complete minimum-mass COPV optimization.

## Explicit propulsion wiring templates

`prop_system.template` selects a YAML file or an inline mapping containing
`circuits`, `nodes` and `branches`. All wiring and design pressures come from
this selection. See [the template schema](Configs/templates/README.md).

Circuit roles bind fluid identities and design flows. Tank nodes reference a
vehicle tank ID; PropSystem injects geometry and fresh initial inventories.
Node `P0` values are pressures, config references or pressure relations to other
nodes. Pump branches reference `pump_id`. Branches identify upstream/downstream
nodes and ports; omitted loss CdA and nozzle throat area are sized automatically.

Multiple supplies feeding the same propellant tank require positive
`demand_fraction` values summing to one. This allocates sizing demand, not runtime
flow. A single supply to each receiving tank defaults to 1.0. Custom parallel
liquid paths require explicit design-flow allocations. The vehicle adapter
currently supports one combustor and one nozzle; unsupported property-transport
cycles are rejected explicitly.

## Electric pump design-point sizing

Pump power is checked during propulsion construction, before the runtime
network is created. The hydraulic branch and network models are unchanged.
Specify each physical pump once:

```yaml
prop_system:
  template: Configs/templates/pump_fed_regulator.yaml
  pumps:
    oxidizer_pump:
      drive: electric
      pressure_rise_pa: 1000000.0  # Pa; optimizer variable
      efficiency: 0.70            # combined pump/motor/controller efficiency
      max_power_kw: 15.0          # per-pump electrical limit
      gas_CdA: 0.0001             # m^2; existing runtime gas-ingestion fallback
    fuel_pump:
      drive: electric
      pressure_rise_pa: 1000000.0
      efficiency: 0.70
      max_power_kw: 15.0
      gas_CdA: 0.0001
```

The selected template references each pump through its branch `pump_id` and
config references in its pressure relations. Inlet and outlet design pressures
must agree with the selected pressure rise. Do not also supply branch `dP`,
`head_model` or `gas_CdA`; these are derived from the physical pump definition.
Nonzero internal pump losses are outside this sizing model. Configure
curves on the physical pump, not as derived `head_model` branch overrides.
Every configured pump must be referenced by exactly one branch. Flow comes
from engine sizing, or the existing explicit branch/circuit `design_mdot`
allocation for custom paths.

Sizing uses inlet **design** pressure and the configured circuit liquid
temperature, assuming no feedline heating. It does not substitute a tank's
optional initial-pressure override for the design inlet pressure. With density
`rho`, `Q = mdot/rho` and `power_kw = pressure_rise_pa * Q / (1000*efficiency)`.
Without a `curve` entry, runtime retains constant pressure rise. With a curve,
the pressure ladder and power calculation still use the design pressure rise,
while the runtime branch uses the scaled quadratic at instantaneous mass flow.

### Scaled quadratic pump curves

`Fluids/pump_curve.py` provides `scaled_pump_curve(mdot, dP, *, reference_mdot,
points=None, coefficients=None)`. Inputs/outputs are kg/s and Pa, not metres
of head. Supply either `[flow, pressure rise]` reference points or descending
polynomial coefficients `[a, b, c]`. Example physical-pump configuration:

```yaml
curve:
  reference_mdot: 2.172
  points: [[0, 3600000], [1.09, 3450000], [1.6, 3160000],
           [2.172, 2548600], [2.8, 1830000], [3.25, 1250000], [4.2, 0]]
```

These are the supplied `Adjustable Pump Curve.xlsx` data with C13 corrected from
345000 to 3450000 Pa; the workbook itself is not changed. The 3200 lbf pump-fed
example uses this reference shape for both pumps, scaled independently.
For reference polynomial `f`, runtime uses
`dP(flow) = design_dP * f(flow * reference_mdot / design_mdot) / f(reference_mdot)`.
Unlike the workbook's unnormalized least-squares fit, this passes exactly through
the design point. A fresh curve is generated on every candidate construction;
no Excel access or curve fitting occurs inside the runtime solve.

The current curve supports decreasing, concave-down quadratics and forward flow
from zero to the first zero-pressure-rise root, capped by the maximum supplied
reference flow when using points. Solver trials may extrapolate the polynomial,
but accepted powered liquid states outside that range raise a model-domain
error. No pressure-rise clipping hides an out-of-domain solution. Gas-fed pumps
retain their existing passive gas restriction. Curve coefficients and maximum
flow appear in `pump_sizing` as `curve_a`, `curve_b`, `curve_c`, `curve_max_mdot`.
This adds neither cavitation/NPSH, motor-speed shutdown logic, nor runtime
electrical power constraints; the power constraint remains design-point only.

`SimResult.pump_sizing[id]` exposes design mass flow (kg/s), volume flow (m^3/s),
inlet density (kg/m^3), inlet P/T (Pa/K), pressure rise (Pa), required/maximum
power (kW) and power margin (kW). `SimResult.constraints["pump.<id>.max_power"]`
is maximum minus required power: zero passes, negative fails. All pump power
margins are collected before rejection. An excessive-power candidate returns
`infeasible_initial_design`, its sizing outputs and negative margins without
running a trajectory. Passing margins remain available in headless results.

These are design-point limits, not runtime electrical limits. Head is not
clipped to satisfy the power rating. Battery energy/mass, motor mass, startup,
cavitation/phase qualification and off-design electrical draw are not modeled.
The migrated pumped example uses illustrative 70% combined efficiency and
15 kW per-pump ratings, which require hardware validation.

## Code boundaries and deferred work

`simulation_types.py` contains cross-system state/result contracts;
`Vehicle/tank_geometry.py` contains shared immutable tank geometry. Domain-local
fluid/material/engine dataclasses remain with their models. Shared state/result
classes are imported directly from `simulation_types`.

Unused analytical aero utilities, section CNa methods, old geometry demo/plotter
and obsolete configs have been removed. Current property-table
generators remain active, explicit offline tools (see `FluidTables/README.md`);
the simulation does not import or execute them.

Real-gas flow equations, RK4 and 3DOF remain deferred. The vertical model projects thrust by cos(AoA);
it does not model lateral motion. Aerodynamic geometry is no longer clipped:
the reader validates the actual candidate against the expanded HDF5 ranges
(length 276–500 in, diameter 8–16 in, nozzle exit 5.9 in to boattail diameter).
Other geometry limits remain in the deck; out-of-range candidates raise errors.
The coefficient hold below Mach 0.1 remains necessary for launch because the
deck still begins at Mach 0.1. Mach above 10 and |AoA| above 15 degrees are
rejected, not extrapolated.

Axial loads now use the model's `ca_distribution`, including engine-on/off
base-pressure effects. The adapter converts inches and per-inch coefficient
densities to SI. Continuous axial and normal densities are integrated as
piecewise-linear functions over the actual vehicle cell boundaries, conserving
force even on nonmatching/nonuniform grids.

The model reports base drag both as a spike already in `dca_dx` and as a point
load. The structural mapping removes `parts["base"]` from the continuous density
and adds the explicit base coefficient once in the final structural cell.
Signed base credits are retained, not clipped. Engine thrust remains at the
forward engine/airframe interface. Distributed inertial forces are subtracted
before cumulative integration into the internal axial load.

When loads are requested, history includes `loads.axial_aero` (external axial
aerodynamic force per cell, N) and `loads.axial` (internal axial load, N,
compression positive). The external force sum is checked against the scalar
flight CA force; it is not added again to that scalar. Aero distributions are
not evaluated by the headless path with load computation disabled.

These distributions use the reader's component-spreading assumptions, not a
resolved surface-pressure field. Nonzero-AoA shares are scaled from the zero-AoA
split, and transonic shares are blended. See `AeroTables/README.md` for the model's
sea-level Reynolds-number and flush-nozzle/fin geometry assumptions.

## Optimizer

Run the 18-variable, minimum-launch-mass search with:

```sh
python3 optimizer.py Configs/optimizer.yaml --output outputs/search-001
# Small integration smoke run (may find no accepted candidate):
python3 optimizer.py Configs/optimizer.yaml --output outputs/smoke-001 --max-evaluations 2
```

Each output directory must be new. The optimizer uses SciPy differential evolution
with a fixed seed, physical-unit bounds, no gradient polishing, and serial
candidate evaluation. SciPy handles variable scaling internally. `popsize` is a
multiplier of the number of free variables; the provided 18-variable configuration
has 72 population members. `max_evaluations` is a hard search budget (including
preflight rejects); one verification flight of the best accepted design is extra.
The supplied 200-evaluation budget is a starting experiment, not evidence of
convergence or global optimality.

`Configs/optimizer.yaml` exposes all bounds, the tank role IDs, search settings,
constraint scales, and evaluation overrides. Its base can be pressure-fed (16
variables) or pump-fed with separate oxidizer and fuel pumps (18 variables). Variables are body diameter, nose fineness,
boattail aft diameter/length, fin span/root/tip/sweep/thickness, chamber pressure,
engine MR, thrust, both pump pressure rises, COPV pressure/volume, and independently
loaded oxidizer/fuel masses. Vehicle length and nozzle exit diameter stay derived.
Nose shape, finish, COPV diameter/wall, engine dimensions/mass, and section input
lengths stay fixed. Changing engine MR does not rebalance loaded propellant.

Aero bounds use the HDF5 deck in metres, with the body diameter minimum raised
to 0.22636 m for COPV packaging and boattail length minimum raised to 0.5715 m
for engine fit. Users can narrow them; settings beyond the deck are rejected. Coupled limits and derived length/
exit limits are checked on each sized vehicle. Propulsion bounds are illustrative
search ranges, not hardware qualification. The optimizer removes COPV mass
overrides and propellant initial-pressure overrides **in its copied config**:
COPV shell mass follows volume/length, and initial feed pressures follow the
chamber-pressure/pump-head ladder. Base files are not modified. Evaluation
overrides extend the flight/AoA horizon to 600 s and the atmosphere table to
300 km. A trajectory exceeding that table stops the search rather than being
accepted using held endpoint atmosphere values.

Scores occupy disjoint classes:

- Feasible completed flights: `mass / (mass + reference_mass)` in [0, 1).
- Completed flights with all required results but constraint violations:
  `1 + V / (1 + V)` in [1, 2).
- Preflight design/geometry rejections: `2 + V / (1 + V)` in [2, 3).
- Incomplete flights, missing required results, or unresolved numerical failures: 3.

Here `V` is the sum of `-margin / scale` for assessed negative required margins.
Exact constraint-name scales override unit scales. They guide ranking within a
class and never relax acceptance thresholds. Redundant tank-fed branch choking
checks remain logged but do not duplicate the tank Pmin penalty. Optional checks
are ignored. The old completion/missing penalty settings are no longer used.
Runtime evaluation failures save individual candidate configs and tracebacks;
configuration failures and identifiable programming errors still raise.

`conditional_geometry: true` maps each DE coordinate into the feasible portion
of its user interval. Thrust bounds enforce the derived nozzle's aero minimum
and available maximum exit diameter; body and boattail diameters then fit that
nozzle and the fixed COPV. Root/tip chords fit the boattail/root. COPV volume and
propellant masses exceed the usable tank-head volumes. An empty interval is an
explicit preflight rejection, not permission to exceed user bounds. Final
vehicle length remains derived and checked against the aero deck. Logged
`variables` are physical values; `search_coordinates` are the original box
coordinates before this conditional mapping. SciPy normalizes its search space.

The pressure-fed SUNDIALS diagnostic run uses isolated workers and resumable
per-candidate files:

```sh
python3 examples/run_optimizer_search.py --output outputs/search-ranked-2000
```

Its configuration is `Configs/optimizer_pressure_fed.yaml`: seed
42, population multiplier 4 (64 members), 2000 candidates, 1-second global steps,
150 km target, 75-second burn limit, regulator control, and fluids stopped at
shutdown while flight coasts. Four independent processes run via a DE map worker;
each has a 300-second wall-time limit. Native crashes/timeouts score 3 and retain
diagnostics. Solver experiments can be installed only in the output directory's
`diagnostic_fixes.py`; source fingerprints prevent an accidental mixed-source
resume. The harness and default simulation use the same SUNDIALS adapter.
Diagnostic replays and winner verification are additional evaluations.

`evaluations.jsonl` records every completed evaluation, its physical variables,
score contributions, constraint metadata, runtime and flight outcome. `search.yaml`
and `base.yaml` capture the resolved inputs. `best.yaml` and `best.json` exist only
if the search found an accepted design. The winner is rerun at
`verification_dt_factor` times the search timestep, with structural load peaks;
`verification.json` records this check and `verified.yaml` exists only if it passes.
`summary.json` distinguishes search acceptance from verification acceptance. If
verification fails, the coarse winner remains available for inspection but is not
reported as a verified accepted design. One finer-timestep check is not a complete
convergence study, and load peaks do not impose structural acceptance limits.

The evaluator reuses loaded property/aero models. It currently rebuilds successful
sizing once for preflight and again inside `simulate`; mutable flight state is
never reused across candidates. COPV walls remain supplied, and pump/battery/motor
mass and structural qualification retain the limitations described above.
