# RASAero II aerodynamic model, team vehicle family

Two files: `dragmodel.h5` (the model) and `dragmodel.py` (the reader). Needs
Python with numpy, scipy and h5py. Nothing else.

It gives, for any vehicle in the team's design range, at any Mach from 0.1
to 10 and any angle of attack from 0 to 15 degrees, for three nose shapes and
three surface finishes:

* drag coefficient, power off and power on
* normal force coefficient
* centre of pressure, fins on the tube or on the boattail
* axial force coefficient
* the normal-force distribution along the body, per component
* the axial-force distribution along the body, per component, with the running axial load

## Use

```python
from dragmodel import DragModel
m = DragModel()

vehicle = dict(omld=12.0, length=360.0, fineness=5.0, exit=9.0, boattail_aft=10.5,
               boattail_length=22.0, span=7.5, root=18.0, tip=6.0, sweep_fraction=0.75,
               thickness=0.375)                                  # inches

m.cd(vehicle, mach=2.0, alpha=5.0)                               # CD, power off
m.cd(vehicle, mach=2.0, alpha=5.0, power_on=True)                # CD with the nozzle firing
m.cn(vehicle, mach=2.0, alpha=5.0)                               # CN
m.cp(vehicle, mach=2.0, alpha=5.0)                               # CP, inches from the nose tip
m.cp(vehicle, mach=2.0, alpha=5.0, fins_on_boattail=True)
m.ca(vehicle, mach=2.0, alpha=5.0)                               # axial force, body axes
m.cd(vehicle, mach=[0.5, 0.9, 1.2, 2.0, 4.0], alpha=0.0)         # arrays broadcast
m.cd(vehicle, mach=2.0, alpha=0.0, nose="ogive", finish="30um")

d = m.cn_distribution(vehicle, mach=2.0, alpha=5.0)              # dCN/dx along the body
d["x"], d["dcn_dx"], d["parts"], d["running_cn"], d["cn"], d["cp"]
s = m.cn_surface(vehicle, alpha=5.0)                             # the same at every Mach: (73, 800)

a = m.ca_distribution(vehicle, mach=2.0, alpha=5.0)              # dCA/dx along the body
a["x"], a["dca_dx"], a["parts"], a["running_ca"], a["ca"]        # running_ca: axial force from the tip back
a["point_loads"]["base"]                                         # (station, load): the base acts at one point
m.ca_distribution(vehicle, mach=2.0, alpha=5.0, power_on=True)   # with the base credit of a firing motor
m.ca_surface(vehicle, alpha=5.0)                                 # the same at every Mach: (73, 800)
m.ca_split_table(vehicle)                                        # RASAero II's alpha-0 drag split per Mach

m.cd_table(vehicle), m.cn_table(vehicle), m.cp_table(vehicle)    # (73 Mach, 4 alpha) grids; m.mach, m.alpha
```

Drag force is `0.5 * rho * V**2 * CD * pi * (omld / 2)**2`; normal force
the same with CN, axial force with CA, and the load per inch along the body
the same with `dcn_dx` or `dca_dx`. Run `python dragmodel.py` to check the file: it re-evaluates
every stored held-out vehicle and prints the error against RASAero II.

## The variables

| Key | Meaning | Range |
|---|---|---|
| `omld` | body diameter, in | 8 to 16 |
| `length` | total length, in | 276 to 500 |
| `fineness` | nose length over diameter | 4 to 6 |
| `boattail_aft` | boattail aft diameter, in | 7.5 to omld |
| `exit` | engine exit diameter, in | 5.9 to boattail_aft |
| `boattail_length` | in | 20 to 25 |
| `span` | exposed semispan, in | **0.5 to 1.2 x omld** |
| `root` | fin root chord, in | 15 to 20 |
| `tip` | fin tip chord, in | 4 to 8 |
| `sweep_fraction` | sweep distance as a fraction of root minus tip | 0.5 to 1.0 |
| `thickness` | fin thickness, in | 0.25 to 0.5 |

Fin span is given in inches like every other length, but its range travels
with the body diameter: from half a diameter to 1.2 diameters. On an 8 in
body that is 4 to 9.6 in, on a 16 in body 8 to 19.2 in. Fin chords and
thickness are plain inch ranges, so fin aspect ratio varies across the
diameter range.

`nose`: `vonkarman`, `ogive`, `conical`. `finish`: `10um` (10.16 micron
camouflage paint, the spec), `20um`, `30um` (30.48 micron rough camouflage).
Values outside the ranges raise an error; the fits are not trusted beyond
them. Finish enters drag only; nose shape enters both drag and normal force.

## How it works

RASAero II's drag is a sum of body terms and fin terms, and the only variable
both read is the diameter. So drag is two smaller models joined by a
reference curve:

    CD = body(D, length, fineness, boattail_aft, boattail_length)
       + fins(D, span, root, tip, sweep, thickness)
       - ref(D)

* `body` is a degree-5 polynomial (252 coefficients per Mach and alpha point)
  fitted to 4,096 RASAero II runs, one per nose shape and finish.
* `fins` is a grid of RASAero II runs, one per finish, read by linear
  interpolation. A polynomial does not work here because RASAero II's fin drag
  charts have discontinuities near Mach 1.1 to 1.4; the grid follows them. The
  levels are not equal across the axes, because the axes do not deserve equal
  resolution: diameter, span and sweep get 7 levels and root, tip and
  thickness 4, which is 21,952 runs. Measured on a full 7-level grid, dropping
  one axis from 7 levels to 4 costs 0.23 points of median CD error for
  diameter, 0.18 for span, 0.10 for sweep, 0.05 for root, 0.03 for tip, and
  nothing at all for thickness, which the fin response is linear in. This grid
  reaches 0.46% median where 5 levels everywhere (15,625 runs) reads 0.75% and
  7 levels everywhere (117,649 runs) reads 0.40%.
* `ref` is RASAero II at the reference vehicle as a function of diameter.
* Every coefficient RASAero II returns is on the vehicle's own area,
  pi D^2 / 4, so it carries a 1/D^2 that has nothing to do with shape. Over
  8 to 16 in that 1/D^2 would dominate the fits: read linearly off a 5-level
  grid it costs up to 3% of CD by itself. So the file stores everything on
  one fixed area, the reference vehicle's (D0 = 12 in): the body fit is of
  CD * (D/D0)^2, the fin grid holds the fin increment over `ref`,
  (fins - ref) * (D/D0)^2, which is nearly flat in diameter, and the reader
  divides (D/D0)^2 back out. You never see the scaled numbers.
* Power on subtracts RASAero II's base-pressure credit,
  `ladder(Mach) * (exit / D)^2 / cos(alpha)`, which is exact to rounding.

Normal force is built the way RASAero II builds it, from parts. RASAero II
gives each part one slope and one station: the nose (whose slope and station
include the afterbody), the boattail, the fin set, the fin's carry-over onto
the body, and one viscous crossflow term on the body's planform. The model
stores those per-part slopes and stations: the body parts from a degree-5
polynomial per nose shape on 4,096 runs, the fin parts from one 5-level grid
with the fin station moved to the vehicle's own length and, for tubes under
13 diameters, the fin and carry-over slopes scaled by the short-tube
correction described below. Slopes are stored on
the fixed area like drag; stations are lengths and are stored as they are.
Then

    CN = (sum of slopes) * alpha + viscous
    CP = (sum of slope * station * alpha + viscous * planform centroid) / CN

which is RASAero II's own arithmetic, so CN and CP are as accurate as the
per-part fits.

The distribution along the body spreads each of those lumps under stated
assumptions, so the curve integrates to the model's CN and its first moment
gives the model's CP. The shape between stations is assumed; the totals are
not.

| Component | How it is spread |
|---|---|
| Nose alone | proportional to dS/dx along the actual nose profile (slender-body theory), tilted to RASAero II's nose-alone CP |
| Body lift | the rest of RASAero II's nose term, a smooth hump over the tube placed to reproduce RASAero II's nose moment |
| Body tube | no potential-flow load, as in RASAero II |
| Boattail | proportional to dS/dx, negative, tilted to RASAero II's boattail CP |
| Viscous crossflow | over the whole body in proportion to local diameter, part by part with RASAero II's planform areas and centroids |
| Fins and carry-over | a smooth hump over the fin root chord with the centroid at RASAero II's station |

### The axial-force distribution

RASAero II reports its drag as a sum of components -- body skin friction,
form drag, nose wave drag, base drag, boattail wave drag and five fin terms
-- and they add up to its total exactly. It never computes a pressure
distribution. So the axial force along the body is RASAero II's own split,
placed where each component acts, the same way the normal force is placed.
The model carries the split (fitted on the same body runs as the drag, the
fins' share from the fin grid), scales it to the model's CA, and spreads each
piece over its part:

| Component | Share at Mach 2, typical | How it is spread |
|---|---|---|
| Body skin friction | about half | over the whole wetted body, in proportion to local circumference times a turbulent skin-friction law, x^-0.2 from the tip |
| Base drag | about a quarter | a point load on the base plane; the power-on credit acts there too |
| Fins (profile, friction, wave, interference, edge) | about an eighth | over the fin root chord, in proportion to one fin's planform area per unit length |
| Nose wave drag | about a tenth, a sixth at Mach 5 | over the nose, in proportion to dS/dx times sin^2 of the surface angle (Newtonian), so it gathers where the nose is steepest |
| Form drag | subsonic only, a few percent | over the nose and boattail, in proportion to their change of area |
| Boattail wave drag | small | over the boattail, in proportion to its change of area |

The curve integrates exactly to the model's CA, and `running_ca`, the axial
force from the nose tip back to each station, ends at CA. The base is a
point load: `point_loads["base"]` gives its station and size, and it also
appears as a one-station spike at the base in `dca_dx` so the integral
closes. For the example vehicle at Mach 2 and alpha 0, a tenth of the axial
force is taken in the first 30 in, half by 300 in, and the base carries 28%.

Three things are assumed rather than given by RASAero II. It gives the split
only at alpha 0, so at angle of attack the alpha-0 split is scaled to the
model's CA. It gives no split inside its transonic fairing, Mach 0.9 to 1.05,
so the Mach 0.9 and 1.05 splits are blended linearly across it. And the shape
within each part (the x^-0.2 friction law, the Newtonian nose weighting, the
planform weighting on the fins) is a stated assumption; the share each part
carries is not. `CA_ASSUMPTIONS` lists them.

Mach is interpolated linearly on the 73-point grid (dense between 0.8 and
1.3). Alpha is interpolated between the four computed angles (0, 5, 10, 15
degrees) with a shape-preserving cubic; the model was checked at those four.

## Accuracy against RASAero II

Worst-case error over the whole Mach grid, per vehicle, on vehicles the
model never saw: 529 for the Von Karman 10 micron stratum and 500 for each
of the other eight, 4,529 in all. `python dragmodel.py` recomputes these.

| | median | 95th percentile | worst |
|---|---|---|---|
| CD, power off, alpha 0 | 0.44 to 0.48% | 1.05 to 1.16% | 1.6 to 2.2% |
| CD, power off, alpha 15 | | 1.00 to 1.12% | |
| CD, power on, alpha 0 | | 1.23 to 1.35% | 1.8 to 2.3% |
| CN, alpha 5 to 15 | | 1.5 to 1.6% | 2.6 to 3.0% |
| CP, all alpha | | 1.5 to 1.7 in | 5.3 to 6.8 in |
| CA split: each component's share of CD | 0.4 points | | 1.5 points |

The CD error is almost all in the fin grid near Mach 1.1 to 1.4, where
RASAero II's fin charts change regime and a grid read linearly cannot follow
them exactly. It roughly doubled when the span range widened from 6 to 9 in
to half a diameter through 1.2 diameters: the same kind of grid now covers a
span band three times wider in relative terms. Refining the grid where it
pays (above) recovered about half of that; going all the way to 7 levels on
every axis would buy another 0.06 points of median for five times the runs.
The distribution along the body integrates to the model's CN and CP by
construction.

**Short, wide bodies are corrected.** Above Mach 1, RASAero II's fin and
carry-over normal-force slopes grow once the cylindrical tube between the
nose and the boattail is shorter than about 12 body diameters (no effect at
13 or more; nose and boattail length themselves do not matter). The fin grid
is solved on the reference vehicle's tube, so on its own it would miss this
by up to 23%. The model carries a correction for it, solved with the port
over tube length, diameter, span, root, tip and sweep (thickness has no
effect) at every Mach, and applies it automatically. Tube length is
`length - fineness * omld - boattail_length`; `m.tube_diameters(vehicle)`
gives it in diameters.

Checked on every one of the 2,557 held-out vehicles with a tube under 13
diameters (2% of the design box):

| tube / diameter | | fin slope, median / worst | CN, worst | CP, worst |
|---|---|---|---|---|
| 9.6 to 11 | uncorrected | 4.1% / 12.8% | 5.8% | 19.1 in |
| | **corrected** | **0.9% / 3.8%** | **1.6%** | **2.8 in** |
| 11 to 13 | either way | 0.7-0.8% / 4.5% | 2.2% | 6.8 in |

The worst held-out vehicle before the correction (diameter 16 in, length
276 in, fineness 6, a 25 in boattail closing to 7.5 in) went from 10% to 0.4%
in CN and from 38 in to 1.6 in in CP. Its steep boattail's negative lift
nearly cancels its small fins, so CP there is a small difference of large
moments and exposes any error in the fin slope.

## Limits

* Every number is a RASAero II prediction. Nothing here is validated against flight.
* The short-tube correction covers normal force. Drag at angle of attack on
  a short tube carries a trace of the same effect, at most 0.65% at alpha 15,
  inside the drag error above; drag at alpha 0 is exact.
* Sea level Reynolds numbers at every Mach. On a real climb the surface finish
  matters less above Mach 2 and vanishes above about Mach 4.
* CD and CN are on the vehicle's own reference area. Compare forces, not
  coefficients, between vehicles of different diameter.
* Inside RASAero II's transonic fairing, Mach 0.9 to 1.05, RASAero II blends
  its two methods linearly in Mach. `cp` does the same; the distribution
  blends the two distributions instead, so its CP can differ from `cp` there
  by under an inch.
* Four fins, hexagonal sharp section, nozzle exit flush with the boattail
  exit plane, no protuberances.
* The design ranges are the team's. Keep the files off public repositories.
