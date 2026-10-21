"""Generate canardmodel.h5 with the reference shock/expansion solver.

Run from the repository root: .venv/bin/python -m AeroTables.generate_canardmodel
The table stores signed-normal-force magnitude before finite-wing corrections.
"""
from pathlib import Path

import h5py
import numpy as np

from Flight.canards import CanardAerodynamics


def main():
    aero = CanardAerodynamics({})
    mach = np.linspace(2.8, 10., 721)  # 0.01 Mach
    angle = np.deg2rad(np.linspace(0., 25., 501))  # 0.05 degrees
    cn = np.array([aero.shock_expansion_cn(np.full_like(angle, m), angle) for m in mach])
    if not np.all(np.isfinite(cn)):
        raise ValueError('Table range crosses the attached-shock domain boundary')
    path = Path(__file__).with_name('canardmodel.h5')
    with h5py.File(path, 'w') as table:
        table.attrs['description'] = 'Shock/expansion CN magnitude; gamma=1.4; no finite-wing correction'
        table.create_dataset('mach', data=mach)
        table.create_dataset('angle_rad', data=angle)
        table.create_dataset('cn', data=cn, compression='gzip')
    print(f'Wrote {path} ({cn.shape[0]} x {cn.shape[1]})')


if __name__ == '__main__':
    main()
