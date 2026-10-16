"""Run six wind-off sensitivity studies from a pressure-fed or pump-fed design."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from sensitivity import SweepResults, make_cases, run_sweep

PSI = 6894.757293168
INCH = 0.0254


def centered_values(baseline, step, count):
    center, delta = Decimal(str(baseline)), Decimal(str(step))
    return [float(center + i * delta) for i in range(-count, count + 1)]


def make_studies(source):
    base = deepcopy(source)
    base['environment']['wind']['enabled'] = False
    base['aero'].setdefault('drag_multiplier', 1.0)
    sections = base['vehicle']['sections']
    mass_paths = []
    for tank_id in ('ox_tank', 'fuel_tank'):
        index = next(i for i, section in enumerate(sections) if section.get('tank_id') == tank_id)
        if any(key in sections[index] for key in ('mass', 'mass_override')):
            raise ValueError(f'{tank_id}: a section mass override would suppress added masses')
        masses = sections[index].setdefault('masses', {})
        if 'sensitivity_addition' in masses:
            raise ValueError('Choose a new addition name: sensitivity_addition already exists')
        masses['sensitivity_addition'] = 0.0
        mass_paths.append(f'vehicle.sections[{index}].masses.sensitivity_addition')
    pump_paths = [f'prop_system.{side}_{segment}_dp'
                  for side in ('fuel', 'ox')
                  for segment in ('pumpin_tank', 'inj_pumpout')]
    if base['prop_system'].get('pumps'):
        # Pressure is continuous even when YAML happens to spell it as an integer.
        for path in pump_paths:
            field = path.split('.')[-1]
            base['prop_system'][field] = float(base['prop_system'][field])
        pressure_study = dict(
            name='system_dp', title='Fuel and LOX pump plumbing pressure drop', parameter='system_dp',
            definition=dict(path=pump_paths[0], linked_paths=pump_paths[1:], offset=True,
                            values=centered_values(0.0, 5 * PSI, 3)),
            divisor=PSI, xlabel='Pressure-drop offset per segment (psi)', unit='psi',
            note='Same -15 to +15 psi offset on tank-to-pump and pump-to-injector losses for fuel and LOX. '
                 'Each segment retains its own baseline. Pump head and injector stiffness stay fixed.')
    else:
        dp = base['prop_system']['ox_tank_inj_dp']
        if dp != base['prop_system']['fuel_tank_inj_dp']:
            raise ValueError('This linked absolute-pressure sweep requires equal baseline system drops')
        pressure_study = dict(
            name='system_dp', title='Fuel and LOX system pressure drop', parameter='system_dp',
            definition=dict(path='prop_system.ox_tank_inj_dp', linked_paths=['prop_system.fuel_tank_inj_dp'],
                            values=centered_values(dp, 5 * PSI, 6)),
            divisor=PSI, xlabel='System pressure drop on each side (psi)', unit='psi',
            note='Both system drops change together by -30 to +30 psi; injector stiffness stays fixed.')
    volume = base['tanks']['press_tank']['volume']
    # Each study keeps all other baseline inputs fixed. Geometry and sizing are
    # recalculated by the normal vehicle model for each changed input.
    studies = [
        dict(name='tank_mass', title='Propellant tank mass', parameter='added_mass_per_tank',
             definition=dict(path=mass_paths[0], linked_paths=mass_paths[1:], start=0.0, stop=20.0, step=2.0),
             divisor=1.0, xlabel='Added mass per propellant tank (kg)', unit='kg per tank',
             note='Same added dry mass on LOX and fuel tanks; COPV unchanged. Total vehicle addition is twice x.'),
        pressure_study,
        dict(name='copv_volume', title='COPV volume', parameter='copv_volume',
             definition=dict(path='tanks.press_tank.volume', values=centered_values(volume, 0.003, 5)),
             divisor=0.001, xlabel='COPV volume (L)', unit='L',
             note='Baseline volume +/-15 L; other inputs fixed, including COPV pressure and diameter.'),
        dict(name='diameter', title='Vehicle diameter', parameter='diameter',
             definition=dict(path='vehicle.OMLD', start=10 * INCH, stop=15 * INCH, step=0.5 * INCH),
             divisor=INCH, xlabel='Vehicle diameter (in)', unit='in',
             note='Fins and COPV outer diameter remain fixed; invalid geometry/aero cases are retained as failures.'),
        dict(name='cstar_efficiency', title='C* efficiency', parameter='cstar_efficiency',
             definition=dict(path='engine.cstar_efficiency', start=0.8, stop=1.0, step=0.025),
             divisor=0.01, xlabel='C* efficiency (%)', unit='percentage point',
             note='Design Pc, MR, Cf efficiency, and thrust target stay fixed; normal engine sizing is applied.'),
        dict(name='drag_multiplier', title='Drag multiplier', parameter='drag_multiplier',
             definition=dict(path='aero.drag_multiplier', start=0.8, stop=1.2, step=0.05),
             divisor=1.0, xlabel='Drag multiplier', unit='multiplier',
             note='Wind-axis drag scaling includes the existing consistent axial structural-load scaling.'),
    ]
    return base, studies


def draw_apogee(results, study, ax):
    key, divisor = study['parameter'], study['divisor']
    points = results.scalar_points(key, 'apogee', include_infeasible=True)
    if points:
        results.plot(key, 'apogee', xscale=divisor, yscale=1000,
                     xlabel=study['xlabel'], ylabel='Apogee (km ASL)',
                     title=study['title'], include_infeasible=True,
                     ax=ax, show=False, save=False, color='#2563eb', linewidth=1.8)
    else:
        ax.set(title=study['title'], xlabel=study['xlabel'], ylabel='Apogee (km ASL)')
        ax.text(.5, .5, 'No completed flights', transform=ax.transAxes, ha='center')
    baseline = next(row for row in results.summary if row['is_baseline'])
    ax.axvline(baseline[key] / divisor, color='#64748b', linestyle='--',
               linewidth=1, label='Baseline')
    all_x = [row[key] / divisor for row in results.summary]
    pad = max((max(all_x) - min(all_x)) * .04, 0.001)
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)


def save_figure(fig, stem):
    for extension in ('.png', '.svg'):
        fig.savefig(stem.with_suffix(extension), dpi=180, bbox_inches='tight')


def plot_suite(folder, studies):
    plots = folder / 'plots'
    plots.mkdir(exist_ok=True)
    overview, axes = plt.subplots(3, 2, figsize=(13, 13), layout='constrained')
    partials, derivative_axes = plt.subplots(3, 2, figsize=(13, 13), layout='constrained')
    summary = []
    for study, overview_ax, derivative_ax in zip(studies, axes.flat, derivative_axes.flat):
        results = SweepResults.load(folder / 'data' / study['name'])
        fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
        draw_apogee(results, study, ax)
        save_figure(fig, plots / study['name'])
        plt.close(fig)
        draw_apogee(results, study, overview_ax)
        try:
            results.plot_derivative(study['parameter'], 'apogee', wrt=study['parameter'],
                xscale=study['divisor'], yscale=1000 / study['divisor'],
                title=study['title'], xlabel=study['xlabel'],
                ylabel=f"Apogee sensitivity (km / {study['unit']})",
                include_infeasible=True, ax=derivative_ax, show=False, save=False,
                color='#7c3aed', marker='o')
        except ValueError as error:
            derivative_ax.set(title=study['title'])
            derivative_ax.text(.5, .5, str(error), transform=derivative_ax.transAxes,
                               ha='center', wrap=True, fontsize=8)
        failed = [dict(case_id=row['case_id'], value=row[study['parameter']],
                       status=row['status'], termination=row.get('termination'),
                       error=row.get('error'),
                       negative_margins={key: value for key, value in row.items()
                           if key.startswith('constraint_records.') and key.endswith('.margin')
                           and isinstance(value, (int, float)) and value < 0})
                  for row in results.summary if row.get('completed') is not True]
        valid = results.scalar_points(study['parameter'], 'apogee', include_infeasible=True)
        summary.append(dict(study=study['name'], note=study['note'],
                            cases=len(results.summary), statuses=dict(Counter(r['status'] for r in results.summary)),
                            apogee_min_km=min((p[1] / 1000 for p in valid), default=None),
                            apogee_max_km=max((p[1] / 1000 for p in valid), default=None),
                            failed_cases=failed))
    save_figure(overview, plots / 'overview')
    save_figure(partials, plots / 'apogee_partials')
    plt.close(overview)
    plt.close(partials)
    (folder / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'Configs/flight_pressure_fed_regulator.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    folder = args.output.resolve()
    if args.plot_only:
        studies = json.loads((folder / 'manifest.json').read_text())['studies']
        plot_suite(folder, studies)
        return
    original = args.config.read_bytes()
    base, studies = make_studies(yaml.safe_load(original))
    specs = []
    for study in studies:
        spec = dict(base_config=str(folder / 'baseline_config.yaml'),
                    sweep=dict(mode='one_at_a_time', workers=args.workers, max_cases=100,
                               ignore_wind=True,
                               parameters={study['parameter']: study['definition']}),
                    outputs=dict(record_history=True, history_detail='standard', compute_loads=False,
                                 directory=str(folder / 'data' / study['name'])))
        parameters, cases = make_cases(base, spec)
        print(f"{study['name']}: {len(cases)} cases including baseline", flush=True)
        specs.append(spec)
    if args.dry_run:
        return
    if folder.exists() and any(folder.iterdir()):
        parser.error('Output directory must be new/empty; use --plot-only to replot saved results')
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'configs').mkdir()
    (folder / 'source_config.yaml').write_bytes(original)
    (folder / 'baseline_config.yaml').write_text(yaml.safe_dump(base, sort_keys=False))
    manifest = dict(source=str(args.config.resolve()), source_sha256=hashlib.sha256(original).hexdigest(),
                    wind_enabled=False, workers=args.workers, studies=studies,
                    feasibility='Design constraints are not assessed. Model-domain and solver errors are recorded per case.')
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for study, spec in zip(studies, specs):
        (folder / 'configs' / f"{study['name']}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        run_sweep(base, spec)
    plot_suite(folder, studies)
    if args.config.read_bytes() != original:
        raise RuntimeError('Source config changed during the run; results use the saved source snapshot')
    print(f'Plots saved to {folder / "plots"}', flush=True)


if __name__ == '__main__':
    main()
