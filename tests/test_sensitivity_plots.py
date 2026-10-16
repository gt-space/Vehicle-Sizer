import csv
import json

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from sensitivity import SweepResults


@pytest.fixture
def saved_sweep(tmp_path):
    # Deliberately omit mode: older configurations rely on the OAT default.
    spec = {'sweep': {'parameters': {
        'pressure': {'values': [1.0, 2.0, 4.0]},
        'mass': {'values': [10.0, 20.0]},
    }}}
    (tmp_path / 'sweep_config.yaml').write_text(yaml.safe_dump(spec))
    cases = [('baseline', 2.0, 10.0, {}),
             ('low', 1.0, 10.0, {'pressure': 1.0}),
             ('high', 4.0, 10.0, {'pressure': 4.0}),
             ('other', 2.0, 20.0, {'mass': 20.0})]
    with (tmp_path / 'cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'overrides'])
        writer.writeheader()
        for case_id, _, _, overrides in cases:
            writer.writerow({'case_id': case_id, 'overrides': json.dumps(overrides)})
    with (tmp_path / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'case_id', 'is_baseline', 'status', 'pressure', 'mass', 'apogee', 'dry_mass'])
        writer.writeheader()
        for case_id, pressure, mass, _ in cases:
            writer.writerow(dict(case_id=case_id, is_baseline=case_id == 'baseline',
                                 status='completed', pressure=pressure, mass=mass,
                                 apogee=pressure ** 2 if mass == 10 else 400,
                                 dry_mass=mass))
    times = np.array([0.0, 1.0, 2.0])
    with h5py.File(tmp_path / 'histories.h5', 'w') as store:
        for case_id, pressure, _, _ in cases:
            group = store.create_group(case_id)
            group.attrs['columns'] = json.dumps(['state.kinematics.t', 'state.kinematics.h', 'state.plant.fluids.propulsion.thrust'])
            group.create_dataset('data', data=np.column_stack([
                times, pressure ** 2 * times, np.full(3, pressure * 100)]))
    return tmp_path


def test_missing_mode_loads_and_isolates_oat_parameter(saved_sweep):
    results = SweepResults.load(saved_sweep)
    assert results.mode == 'one_at_a_time'
    assert results.scalar_points('pressure', 'apogee') == [
        (1.0, 1.0, 'low'), (2.0, 4.0, 'baseline'), (4.0, 16.0, 'high')]
    assert results.field_catalog(contains='apogee')[0]['category'] == 'Flight performance'
    assert results.history_fields() == ['state.kinematics.h', 'state.kinematics.t', 'state.plant.fluids.propulsion.thrust']


def test_full_grid_requires_filters(saved_sweep):
    path = saved_sweep / 'sweep_config.yaml'
    spec = yaml.safe_load(path.read_text())
    spec['sweep']['mode'] = 'full_grid'
    path.write_text(yaml.safe_dump(spec))
    results = SweepResults.load(saved_sweep)
    with pytest.raises(ValueError, match='needs filters'):
        results.scalar_points('pressure', 'apogee')
    assert results.scalar_points('pressure', 'apogee', filters={'mass': 10}) == [
        (1.0, 1.0, 'low'), (4.0, 16.0, 'high')]


def test_invalid_mode_is_rejected(saved_sweep):
    path = saved_sweep / 'sweep_config.yaml'
    spec = yaml.safe_load(path.read_text())
    spec['sweep']['mode'] = 'typo'
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match='sweep.mode'):
        SweepResults.load(saved_sweep)


def test_derivatives_on_nonuniform_sweep(saved_sweep):
    results = SweepResults.load(saved_sweep)
    x, derivative = results.scalar_derivative('apogee', 'pressure')
    np.testing.assert_allclose(derivative, 2 * x)
    x, time, derivative = results.history_derivative('state.kinematics.h', 'pressure')
    np.testing.assert_allclose(derivative, 2 * x[:, None] * time[None, :])


@pytest.mark.parametrize('method,arguments,options', [
    ('plot', ('pressure', 'apogee'), {'y2': 'dry_mass'}),
    ('plot_history', ('state.plant.fluids.propulsion.thrust',), {'group_by': 'pressure'}),
    ('plot_case_history', (), {'y': 'state.plant.fluids.propulsion.thrust', 'y2': 'state.kinematics.h'}),
    ('plot_derivative', ('pressure', 'apogee'), {'wrt': 'pressure'}),
    ('plot_history_derivative', ('state.kinematics.h',), {'wrt': 'pressure'}),
])
def test_plot_saved_results(saved_sweep, method, arguments, options):
    results = SweepResults.load(saved_sweep)
    output = saved_sweep / f'{method}.png'
    figure = getattr(results, method)(*arguments, **options, save=output, show=False)
    try:
        assert output.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        if method == 'plot':
            assert len(figure.axes) == 2
            np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [1, 4, 16])
    finally:
        plt.close(figure)


def test_cli_lists_fields_and_writes_svg(saved_sweep, capsys):
    from reporting.plot_sensitivities import main
    main([str(saved_sweep), '--list-fields'])
    assert 'state.kinematics.h' in capsys.readouterr().out
    output = saved_sweep / 'cli.svg'
    main([str(saved_sweep), '--x', 'pressure', '--y', 'apogee',
          '--y2', 'dry_mass', '--output', str(output)])
    assert '<svg' in output.read_text()
