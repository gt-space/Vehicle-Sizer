"""Run a config and plot station-resolved structural loads during powered flight.

Example: python -m reporting.plot_peak_loads --config Configs/flight_pressure_fed_regulator.yaml
"""
from __future__ import annotations
from simulation_types import history_samples


import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

from Configs.loader import load_config
from simulation import simulate

QUANTITIES = (('axial', 'Axial force', 'kN'),
              ('bending', 'Bending moment', 'kN·m'),
              ('shear', 'Shear force', 'kN'))


def summarize_loads(samples):
    """Distinguish station-wise magnitude envelopes from simultaneous profiles."""
    if not samples:
        raise ValueError('No powered-flight load samples available')
    station = np.asarray(samples[0]['loads']['station'])
    if any(not np.array_equal(s['loads']['station'], station) for s in samples):
        raise ValueError('Load samples must share the same axial stations')
    time = np.array([s['kinematics'].t for s in samples])
    matrices, peaks, envelopes, envelope_times = {}, {}, {}, {}
    for key, _, _ in QUANTITIES:
        matrix = np.stack([s['loads'][key] for s in samples])
        if matrix.shape != (len(samples), len(station)) or not np.all(np.isfinite(matrix)):
            raise ValueError(f'Invalid {key} profiles')
        matrices[key] = matrix
        peak_t, peak_x = np.unravel_index(np.abs(matrix).argmax(), matrix.shape)
        peaks[key] = dict(time_s=float(time[peak_t]), station_m=float(station[peak_x]),
                          signed_value_SI=float(matrix[peak_t, peak_x]), sample_index=int(peak_t))
        indices = np.abs(matrix).argmax(axis=0)
        envelopes[key] = np.abs(matrix[indices, np.arange(len(station))])
        envelope_times[key] = time[indices]
    return station, time, matrices, peaks, envelopes, envelope_times


def plot_loads(result, output):
    output.mkdir(parents=True, exist_ok=True)
    full_samples = [s for s in history_samples(result) if 'loads' in s]
    samples = [s for s in full_samples if s['engine_on']]
    station, time, matrices, peaks, envelopes, envelope_times = summarize_loads(samples)
    # Bending supplies an explicit criterion for a single coherent load case;
    # force and moment cannot be compared numerically across unlike units.
    selected = peaks['bending']['sample_index']
    sections = [s for key, s in result.design_summary['section'].items() if key != 'engine']
    for mode in ('burn_envelope', 'peak_bending_snapshot'):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, layout='constrained')
        for ax, (key, title, unit) in zip(axes, QUANTITIES):
            if mode == 'burn_envelope':
                values = envelopes[key]
                label = 'Maximum magnitude at each station over burn'
            else:
                values = matrices[key][selected]
                label = f"At peak bending: t = {time[selected]:.3f} s"
            ax.plot(station, values / 1000, color='#2563eb', linewidth=1.7, label=label)
            ax.set_ylabel(f'{title} ({unit})')
            ax.axhline(0, color='#64748b', linewidth=.6)
            ax.grid(alpha=.2)
            ax.legend(fontsize=9, loc='best')
            for section in sections:
                ax.axvline(section['station'], color='#94a3b8', linestyle=':', linewidth=.7)
            ax.set_xlim(0, result.design_summary['length'])
        for section in sections:
            axes[0].text(section['station'] + section['length'] / 2, 1.02, section['label'], rotation=45,
                         transform=axes[0].get_xaxis_transform(), ha='left', va='bottom', fontsize=8)
        axes[-1].set_xlabel('Axial station from nose tip (m) → aft')
        for extension in ('png', 'svg'):
            fig.savefig(output / f'{mode}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    columns = {'station_m': station}
    for key, _, _ in QUANTITIES:
        columns[f'{key}_max_abs_SI'] = envelopes[key]
        columns[f'{key}_envelope_time_s'] = envelope_times[key]
        columns[f'{key}_at_peak_bending_SI'] = matrices[key][selected]
    with (output / 'load_profiles.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    np.savez_compressed(output / 'burn_load_history.npz', station_m=station, time_s=time, **matrices)
    # Keep event-side samples distinct even when their timestamps coincide.
    with (output / 'load_history.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('sample_index', 'time_s', 'engine_on', 'station_m',
                         'axial_N', 'shear_N', 'bending_N_m'))
        for index, sample in enumerate(full_samples):
            loads = sample['loads']
            writer.writerows(
                (index, sample['kinematics'].t, sample['engine_on'], x, axial, shear, bending)
                for x, axial, shear, bending in zip(
                    loads['station'], loads['axial'], loads['shear'], loads['bending'], strict=True))
    summary = dict(burn_duration_s=result.burn_duration, burn_complete=result.burn_complete,
                   termination=result.termination, sample_count=len(samples), peaks=peaks,
                   full_history_sample_count=len(full_samples),
                   full_history_csv='load_history.csv',
                   snapshot_criterion='Maximum absolute bending moment anywhere along the vehicle during burn',
                   snapshot_time_s=float(time[selected]),
                   snapshot_dynamic_pressure_Pa=float(samples[selected]['atmosphere'].q),
                   axial_sign='Positive compression; negative tension',
                   envelope_definition='Maximum absolute value at each station over recorded powered samples',
                   sampling='Initial state, accepted endpoints, and powered side of cutoff; no time downsampling')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('Configs/flight_pressure_fed_regulator.yaml'))
    parser.add_argument('--output', type=Path, default=Path('outputs/pressure_fed_loads'))
    parser.add_argument('--wind-off', action='store_true')
    parser.add_argument('--dt', type=float, help='Override flight sampling step in seconds')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a new/empty output directory')
    original = args.config.read_bytes()
    cfg = load_config(args.config)
    if args.wind_off:
        cfg['environment']['wind']['enabled'] = False
    if args.dt is not None:
        cfg['simulation']['dt'] = args.dt
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'source_config.yaml').write_bytes(original)
    (args.output / 'run_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    print('Running with structural loads enabled...', flush=True)
    result = simulate(cfg, record_history=True, compute_loads=True)
    summary = plot_loads(result, args.output)
    summary.update(source_config=str(args.config.resolve()),
                   source_sha256=hashlib.sha256(original).hexdigest(),
                   wind_enabled=cfg['environment']['wind']['enabled'],
                   dt_s=cfg['simulation']['dt'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    if args.config.read_bytes() != original:
        raise RuntimeError('Source config changed during execution')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Plots saved to {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
