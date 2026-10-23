"""Generate shock and forward roll-response tables from the shared canard model.

"""
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Support direct script execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Flight.canards import CanardAerodynamics, LEADING_EDGE_DEG


# Reference geometry: the shared model derives a triangular, fixed-sweep shape.
# Root/diameter scaling remains live; the roll table depends on shape/aspect ratio.
ROOT_CHORD_M = 0.25
BODY_DIAMETER_M = 0.31115

SHOCK_MACH = np.linspace(2.8, 10., 721)
SHOCK_ANGLE_RAD = np.deg2rad(np.linspace(0., 25., 501))
ROLL_MACH = np.linspace(0., 10., 401)
BODY_ALPHA_RAD = np.deg2rad(np.linspace(0., 15., 151))
DEFLECTION_RAD = np.deg2rad(np.linspace(0., 10., 201))


def main():
    aero = CanardAerodynamics({})
    cn = np.array([aero.shock_expansion_cn(np.full_like(SHOCK_ANGLE_RAD, m), SHOCK_ANGLE_RAD)
                   for m in SHOCK_MACH])
    if not np.all(np.isfinite(cn)):
        raise ValueError('Shock grid crosses the attached-shock model domain')
    # Use freshly generated data, never an existing HDF5 file or cached lookup.
    lookup = RegularGridInterpolator((SHOCK_MACH, SHOCK_ANGLE_RAD), cn,
                                     bounds_error=False, fill_value=np.nan)
    aero = CanardAerodynamics({}, shock_lookup=lookup)
    geometry = aero.surface_geometry(ROOT_CHORD_M, BODY_DIAMETER_M / 2)
    path = Path(__file__).with_suffix('.h5')
    temporary = path.with_suffix('.h5.tmp')
    try:
        with h5py.File(temporary, 'w') as table:
            table.attrs.update(format_version=2, model_version=1, gamma=1.4,
                               description='Shock/expansion CN and forward canard roll response')
            for name, data in [('mach', SHOCK_MACH), ('angle_rad', SHOCK_ANGLE_RAD), ('cn', cn)]:
                table.create_dataset(name, data=data, compression='gzip')
            roll = table.create_group('roll')
            roll.attrs.update(planform='triangular', leading_edge_deg=LEADING_EDGE_DEG,
                              aspect_ratio=geometry.aspect_ratio, root_chord_m=ROOT_CHORD_M,
                              body_diameter_m=BODY_DIAMETER_M)
            for name, data in [('mach', ROLL_MACH), ('alpha_rad', BODY_ALPHA_RAD),
                               ('deflection_rad', DEFLECTION_RAD)]:
                roll.create_dataset(name, data=data)
            values = roll.create_dataset('coefficient',
                shape=(len(ROLL_MACH), len(BODY_ALPHA_RAD), len(DEFLECTION_RAD)),
                dtype='f8', compression='gzip')
            for i, mach in enumerate(ROLL_MACH):
                curve = aero.roll_coefficient(geometry, mach, DEFLECTION_RAD[None,:],
                                               BODY_ALPHA_RAD[:,None])
                if not np.all(np.isfinite(curve)) or np.any(np.diff(curve, axis=1) < 0):
                    raise ValueError(f'Invalid or nonmonotonic roll response at Mach {mach:g}')
                values[i] = curve
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Wrote {path}: shock {cn.shape}, roll '
          f'{(len(ROLL_MACH), len(BODY_ALPHA_RAD), len(DEFLECTION_RAD))}')


if __name__ == '__main__':
    main()
