"""Wind sweeps relative to launch altitude; no production files are modified.
Run: .venv/bin/python wind_sensitivity/run.py --workers 3
Replot existing results: .venv/bin/python wind_sensitivity/run.py --plot-only
"""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import json
import multiprocessing
import os
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
CUTOFFS = [0, 5, 10, 15, 20, 30, 40, 55, 75, 100, 150, 200, 300, 400, 550, 750, 1000, 2000, 3000, 6500, 10000, 30000, 100000]
BANDS = [1000, 10000]
PERCENTAGES = list(range(100, 29, -10))


def run_case(case):
    from Configs.loader import load_config
    from Flight.environment import WindProfile
    from simulation import simulate
    from diagnostics.errors import EvaluationFailure
    cfg = deepcopy(load_config(OUT / 'config.yaml'))
    cfg['environment']['wind'].update(enabled=True, profile=str(OUT / 'wind_profile.csv'))
    elevation = float(cfg['launch']['altitude'])
    original = WindProfile.wind

    def modified(self, altitude):
        wind = original(self, altitude)
        height = altitude - elevation
        if case['mode'] == 'cutoff':
            factor = 1.0 if height < case['height_agl_m'] else 0.0
        elif case['mode'] == 'scale':
            factor = case['wind_percent'] / 100 if height < case['height_agl_m'] else 1.0
        else:
            factor = 1.0
        return tuple(value * factor for value in wind)

    WindProfile.wind = modified
    row = dict(case)
    start = time.perf_counter()
    try:
        r = simulate(cfg)
        row.update(status=r.termination, downrange_km=r.final_x / 1000,
                   absolute_downrange_km=abs(r.final_x) / 1000, apogee_km=r.apogee / 1000,
                   apogee_time_s=r.apogee_time, min_stability=r.min_stability_calibers,
                   max_aoa_deg=r.max_aoa_deg, burn_s=r.burn_duration)
    except EvaluationFailure as error:
        row.update(status='failed', error=str(error))
        (OUT / f"failure_{case['id']}.json").write_text(json.dumps(error.diagnostic, indent=2))
    finally:
        WindProfile.wind = original
    row['runtime_s'] = time.perf_counter() - start
    return row


def save(rows):
    rows = sorted(rows, key=lambda row: row['order'])
    (OUT / 'results.json').write_text(json.dumps(rows, indent=2))
    with (OUT / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def plot_results():
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/vehicle-sizer-wind-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows = json.loads((OUT / 'results.json').read_text())
    baseline = next(r for r in rows if r['mode'] == 'baseline')
    cuts = sorted((r for r in rows if r['mode'] == 'cutoff' and r['height_agl_m'] in CUTOFFS), key=lambda r: r['height_agl_m'])
    metrics = [('downrange_km', 'Signed downrange at apogee (km)'), ('apogee_km', 'Apogee altitude MSL (km)')]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    successful = [r for r in cuts if r['status'] == 'apogee']
    heights = [r['height_agl_m'] for r in successful]
    cutoff_metrics = [('downrange_km', 'Downrange at Apogee (km)'), ('apogee_km', 'Apogee (km)')]
    for ax, (metric, ylabel) in zip(axes, cutoff_metrics):
        # Connect valid samples across failures without inventing sampled values.
        ax.plot(heights, [r[metric] for r in successful], 'o-', color='#146d9c')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
    axes[-1].set_xscale('symlog', linthresh=10)
    axes[-1].set_xticks([0, 10, 100, 1000, 10000, 100000],
                       ['0 m', '10 m', '100 m', '1 km', '10 km', '100 km'])
    axes[-1].set_xlabel('Wind Cutoff point')
    fig.suptitle('Wind Cutoff Sensitivity')
    fig.tight_layout()
    fig.savefig(OUT / 'wind_cutoff.png', dpi=180)
    fig.savefig(OUT / 'wind_cutoff.svg')
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for band, color in zip(BANDS, ['#146d9c', '#d47722']):
        selected = [dict(baseline, wind_percent=100)] + sorted(
            (r for r in rows if r['mode'] == 'scale' and r['height_agl_m'] == band),
            key=lambda r: -r['wind_percent'])
        for ax, (metric, ylabel) in zip(axes, metrics):
            ax.plot([r['wind_percent'] for r in selected], [r.get(metric, np.nan) for r in selected],
                    'o-', color=color, label=f'Below {band / 1000:g} km AGL')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.25)
            for r in selected:
                if r['status'] != 'apogee':
                    ax.text(r['wind_percent'], .05 if band == 1000 else .17,
                            f'Failed ({band / 1000:g} km)', rotation=90, ha='center',
                            color=color, transform=ax.get_xaxis_transform())
    axes[0].legend()
    axes[-1].set_xticks(PERCENTAGES)
    axes[-1].set_xlim(102, 28)
    axes[-1].set_xlabel('Wind retained within altitude band (%) — wind above band unchanged')
    fig.suptitle('Low-altitude wind scaling sensitivity')
    fig.tight_layout()
    fig.savefig(OUT / 'wind_scaling.png', dpi=180)
    fig.savefig(OUT / 'wind_scaling.svg')
    plt.close(fig)

    text = ['# Wind sensitivity', '',
            'Heights are above launch (AGL). Apogee is MSL; downrange is signed displacement at apogee, not landing range.',
            'Both wind components are scaled together. Cutoff cases set wind to zero above the cutoff; scaling cases retain the original profile above the selected band.',
            'Cutoff plot uses a linear axis from 0–10 m and logarithmic spacing above 10 m. Apogee is absolute MSL altitude.',
            'Transitions at band boundaries are abrupt. The cutoff plot connects successful samples across failed regions; markers represent successful simulations only. Raw failures and older cutoff results remain in the results files. A 100% scaling case is identical to the full-profile baseline and is reused.', '',
            '| Case | Downrange (km) | Apogee (km MSL) | Status |', '|---|---:|---:|---|']
    for r in sorted(rows, key=lambda r: r['order']):
        text.append(f"| {r['id']} | {r.get('downrange_km', float('nan')):.3f} | {r.get('apogee_km', float('nan')):.3f} | {r['status']} |")
    text += ['', 'These sweeps show cumulative and trajectory-dependent effects; differences between cutoff cases are not independent additive contributions from altitude layers.', '',
             'Reproduce: `.venv/bin/python wind_sensitivity/run.py --workers 3`',
             'Replot: `.venv/bin/python wind_sensitivity/run.py --plot-only`', '',
             'Inputs: `config.yaml`, `wind_profile.csv`. Outputs: `results.csv`, `results.json`, PNG/SVG plots, and failure diagnostics.']
    (OUT / 'README.md').write_text('\n'.join(text) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse saved cases; run missing cases only')
    args = parser.parse_args()
    if not args.plot_only:
        if args.workers < 1:
            parser.error('--workers must be positive')
        cases = [dict(id='full_profile', mode='baseline', height_agl_m=None, wind_percent=100)]
        cases += [dict(id=f'cutoff_{h}m', mode='cutoff', height_agl_m=h, wind_percent=100) for h in CUTOFFS]
        cases += [dict(id=f'scale_{h}m_{pct}pct', mode='scale', height_agl_m=h, wind_percent=pct)
                  for h in BANDS for pct in PERCENTAGES if pct != 100]
        rows = json.loads((OUT / 'results.json').read_text()) if args.resume else []
        order = {case['id']: i for i, case in enumerate(cases)}
        for row in rows:
            row['order'] = order.get(row['id'], len(cases) + (row.get('height_agl_m') or 0))
        completed = {row['id'] for row in rows}
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(run_case, dict(case, order=i)) for i, case in enumerate(cases) if case['id'] not in completed]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                save(rows)
                print(json.dumps(row), flush=True)
    plot_results()
