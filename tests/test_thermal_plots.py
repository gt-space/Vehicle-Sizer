from types import SimpleNamespace
import numpy as np
import pytest
import flight_plots


def state(t, thermal=True):
    output = SimpleNamespace(node={'tank': {'cells': {
        'station': np.array([0., 1.]), 'wall_T': np.array([290.+t, 300.+t])}}}) if thermal else None
    return {'kinematics': SimpleNamespace(t=t), 'plant': SimpleNamespace(thermal=output)}


@pytest.mark.parametrize('history', [
    [state(0), state(10), state(20, False)],
    [state(0, False), state(10), state(20), state(30, False)],
    [state(0, False), state(10, False)],
    [state(0), state(10, False)],
])
def test_missing_thermal_history_renders(tmp_path, history):
    path = tmp_path / 'thermal.png'
    flight_plots.plot_thermal_history(history, path, burnout=15.)
    assert path.stat().st_size > 0


def test_plot_stops_at_last_thermal_sample(monkeypatch):
    def inspect(figure, axes, path):
        map_axis, profile_axis = axes
        assert map_axis.get_xlim() == (0., 10.)
        assert 'last sample at t = 10 s' in map_axis.texts[0].get_text()
        assert [line.get_label() for line in profile_axis.lines] == ['t = 0 s', 't = 10 s']
        np.testing.assert_allclose(profile_axis.lines[-1].get_ydata(), [300., 310.])
        flight_plots.plt.close(figure)
    monkeypatch.setattr(flight_plots, '_finish', inspect)
    flight_plots.plot_thermal_history([state(0), state(10), state(20, False)], 'unused', burnout=15.)
