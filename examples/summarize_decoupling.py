"""Summarize completed isolated benchmark runs and draw a standalone figure."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/decoupling_benchmark'


def main():
    reports = {}
    groups = defaultdict(list)
    for file in sorted(OUT.glob('*.json')):
        d = json.loads(file.read_text())
        if 'mode' not in d or not d.get('success'):
            continue
        reports[file.stem] = d
        dt = d['coupling_dt'] if d['mode'] != 'baseline' else d['baseline_dt']
        groups[(d['mode'], dt, d.get('rk_rtol'))].append(d)
    baseline = groups[('baseline', 2., 1e-7)]
    baseline_time = statistics.median(d['wall_s'] for d in baseline)
    reference_name = 'baseline_dt0p05'
    reference = reports[reference_name]['metrics']
    rows = []
    for (mode, dt, rtol), runs in groups.items():
        d = runs[-1]
        m = d['metrics']
        wall = statistics.median(x['wall_s'] for x in runs)
        flight = statistics.median(x['timings']['flight']['wall_s'] if mode != 'decoupled'
                                    else x['harness']['flight_wall_s'] for x in runs)
        rows.append(dict(mode=mode, dt=dt, rtol=rtol, runs=len(runs), wall_s=wall,
                         min_wall_s=min(x['wall_s'] for x in runs), max_wall_s=max(x['wall_s'] for x in runs),
                         flight_wall_s=flight, speedup=baseline_time/wall,
                         fluid_updates=d['timings']['fluid_update']['calls'],
                         fluid_restores=d['timings'].get('fluid_restore',{}).get('calls',0),
                         apogee_m=m['apogee'],
                         apogee_delta_reference_pct=100*(m['apogee']/reference['apogee']-1),
                         burnout_s=m['burn_duration'],
                         burnout_delta_reference_ms=1000*(m['burn_duration']-reference['burn_duration']),
                         final_mass_kg=m['final_mass'], max_q_Pa=m['max_q'],
                         final_pitch_deg=math.degrees(m['final_pitch_angle']),
                         rail_exit_s=m['rail_exit_time']))
    rows.sort(key=lambda x: (x['mode'] != 'baseline', -x['dt'], x['rtol']))
    with (OUT/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    hashed = [d['source_hashes'] for d in reports.values() if 'source_hashes' in d]
    same_sources = all(h == hashed[0] for h in hashed)
    summary = dict(reference=reference_name, production_source_hashes_match=same_sources,
                   cases=rows)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    text = [
        'Pressure-fed regulator, RAP_HWM wind enabled: isolated decoupling experiment',
        '',
        'Original and experimental harness = archived commit 5b84921; production = current split solver.',
        'All modes: history enabled, structural loads/plots/constraint assessment disabled.',
        'Times include model setup and simulation, plus small CSV serialization overhead;',
        'they exclude Python/import startup. Runs use fresh processes and execute serially.',
        'Lightweight identical timing wrappers are installed in both modes.',
        '',
        'Harness: unchanged implicit IDAS fluid physics/tolerances; explicit adaptive RK45',
        '(Dormand-Prince 5(4), not classical fixed-step RK4), normally rtol=1e-7, max step=2 s.',
        'Additional tight check: rtol=1e-9, max flight step=0.5 s, coupling interval=0.5 s.',
        'Absolute tolerances: [1e-5 m, 1e-5 m, 1e-6 m/s, 1e-6 m/s, 1e-9 rad, 1e-9 rad/s].',
        'Piecewise-linear thrust and fixed-origin mass-moment interpolation, split at fluid events.',
        'Fluid boundary prediction: endpoint pressure and initial axial force, as in baseline.',
        'Coupling interval is fixed per benchmark case; no outer boundary-correction loop.',
        'Flight error control is adaptive. Coupling accuracy is evaluated by interval refinement.',
        'All configured tank thermal models are None: external thermal coupling is untested.',
        'Harness rejects external heating, post-shutdown fluid evolution, and structural loads.',
        'Powered-apogee rollback is outside scope. Actual runs reach apogee after shutdown.',
        'Intermediate node thermodynamics are not interpolated; exported CSVs contain flight quantities only.',
        '',
        f'Production source fingerprints agree across instrumented runs: {same_sources}',
        f'Reference for deltas: original solver with dt=0.05 s ({reference_name}).',
        'This is a refined numerical comparison, not a certified exact solution.',
        '',
        'mode       dt[s]   RK rtol  n  total[s]  flight[s] speedup fluid calls apogee[m] delta[%] burnout[s] final pitch[deg]',
    ]
    for r in rows:
        text.append(f"{r['mode']:10} {r['dt']:5g} {r['rtol']:9g} {r['runs']:2d} {r['wall_s']:9.3f} {r['flight_wall_s']:9.3f} "
                    f"{r['speedup']:7.2f} {r['fluid_updates']:11d} {r['apogee_m']:10.3f} "
                    f"{r['apogee_delta_reference_pct']:8.4f} {r['burnout_s']:10.6f} {r['final_pitch_deg']:10.5f}")
    original = reports['baseline_3']
    candidate = reports['decoupled_dt0p5_3']
    tight = reports['decoupled_dt0p5_tight']
    text.extend(['', 'Measured diagnostics:',
                 f"Original flight attempts: {original['timings']['flight_attempt']}",
                 f"Original fluid restores: {original['timings']['fluid_restore']['calls']}",
                 f"Harness at 0.5 s: {candidate['harness']['rk_accepted_steps']} accepted RK steps, "
                 f"{candidate['harness']['rk_rejected_steps']} error-control rejections, zero fluid restores.",
                 f"Harness fluid events: {candidate['harness']['fluid_events']}",
                 'Final pitch remains sensitive to the original solver timestep: '
                 f"{math.degrees(reports['baseline_dt0p1']['metrics']['final_pitch_angle']):.5f} deg at dt=0.1 s, "
                 f"{math.degrees(reference['final_pitch_angle']):.5f} deg at dt=0.05 s.",
                 'Tightening harness RK controls at the same coupling interval changes apogee by '
                 f"{tight['metrics']['apogee']-candidate['metrics']['apogee']:.6f} m and final pitch by "
                 f"{math.degrees(tight['metrics']['final_pitch_angle']-candidate['metrics']['final_pitch_angle']):.6f} deg.",
                 'At coupling dt=0.05 s the harness is about as expensive as the original dt=0.05 s;',
                 'the speed advantage depends on retaining larger coupling intervals than flight steps.',
                 '', 'Reproduction:',
                 '.venv/bin/python examples/benchmark_decoupling.py --mode baseline --output outputs/decoupling_benchmark/new_baseline.json',
                 '.venv/bin/python examples/benchmark_decoupling.py --mode decoupled --coupling-dt 0.5 --flight-max-step 2 --output outputs/decoupling_benchmark/new_decoupled.json',
                 'Run the baseline/harness commands above from archived commit 5b84921.',
                 '.venv/bin/python examples/benchmark_decoupling.py --mode production --coupling-dt 0.5 --flight-max-step 2 --output outputs/decoupling_benchmark/production.json',
                 '.venv/bin/python examples/summarize_decoupling.py'])
    (OUT/'report.txt').write_text('\n'.join(text)+'\n')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    selected = [r for r in rows if r['rtol']==1e-7 and ((r['mode']=='baseline' and r['dt'] in (2., .1, .05))
                or (r['mode'] in ('decoupled', 'production') and r['dt'] in (2., .5, .25)))]
    labels = [f"{'Original' if r['mode']=='baseline' else r['mode'].capitalize()}\n{r['dt']:g} s" for r in selected]
    colors = ['#63738a' if r['mode']=='baseline' else '#168b8a' for r in selected]
    ax = axes[0,0]
    bars = ax.bar(labels, [r['wall_s'] for r in selected], color=colors)
    ax.bar_label(bars, labels=[f"{r['wall_s']:.1f}s" for r in selected], padding=3)
    ax.set(ylabel='Wall time [s]', title='Runtime: median where repeated')
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylim(0, baseline_time*1.15)
    for name, label, color in [('baseline_3','Original, dt=2 s','#63738a'),
                              ('baseline_dt0p05','Original, dt=0.05 s','#b97726'),
                              ('decoupled_dt0p5_3','Decoupled, interval=0.5 s','#168b8a')]:
        data = np.genfromtxt(OUT/f'{name}.csv', delimiter=',', names=True)
        axes[0,1].plot(data['t'], data['h']/1000, label=label, color=color, linewidth=1.4)
        axes[1,1].plot(data['t'], np.degrees(data['theta']), label=label, color=color, linewidth=1.2)
    axes[0,1].set(xlabel='Time [s]', ylabel='Altitude [km]', title='Trajectory comparison')
    axes[0,1].legend(fontsize=8)
    axes[1,1].set(xlabel='Time [s]', ylabel='Pitch [deg]', title='Pitch: accuracy matters beyond apogee')
    ax = axes[1,0]
    decoupled = sorted((r for r in rows if r['mode']=='decoupled' and r['rtol']==1e-7), key=lambda r:r['dt'])
    ax.semilogx([r['dt'] for r in decoupled], [r['apogee_m']/1000 for r in decoupled], 'o-', color='#168b8a', label='Decoupled')
    ax.axhline(reference['apogee']/1000, color='#b97726', linestyle='--', label='Original dt=0.05 s')
    ax.axhline(baseline[0]['metrics']['apogee']/1000, color='#63738a', linestyle=':', label='Original dt=2 s')
    ax.set(xlabel='Fluid coupling interval [s]', ylabel='Apogee [km]', title='Coupling-interval refinement')
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Pressure-fed rocket with wind: decoupling benchmark', fontsize=15)
    fig.savefig(OUT/'comparison.png', dpi=180)
    plt.close(fig)
    print('\n'.join(text))


if __name__ == '__main__':
    main()
