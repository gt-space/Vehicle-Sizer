import pytest

import main
import csv
from types import SimpleNamespace
from copy import deepcopy
from constraints import finalize
from simulation_types import SimResult


@pytest.mark.parametrize('enforce', [False, True])
def test_manual_run_ignores_search_limits_unless_requested(monkeypatch, enforce):
    cfg = {'constraints': {'max_length_to_diameter': 25., 'goal_apogee': 150000.},
           'tanks': {'press_tank': {'min_temperature': 190.}}, 'simulation': {}}
    before = deepcopy(cfg)
    monkeypatch.setattr('sys.argv', ['main.py', 'candidate.yaml'] + (['--enforce-constraints'] if enforce else []))
    monkeypatch.setattr(main, 'load_config', lambda path: cfg)

    def simulate(candidate, **kwargs):
        assert ('constraints' in candidate) is enforce
        assert candidate['tanks'] == cfg['tanks']
        raise RuntimeError('checked manual policy')

    monkeypatch.setattr(main, 'simulate', simulate)
    with pytest.raises(RuntimeError, match='checked manual policy'):
        main.main()
    assert cfg == before


@pytest.mark.parametrize("termination", ["infeasible_initial_design", "infeasible_operating_state"])
def test_cli_reports_rejection_before_history_without_writing_outputs(monkeypatch, termination):
    result = finalize(SimResult(
        termination=termination,
        constraints={"pump.oxidizer_pump.max_power": -0.15689966065800753},
    ), {"goal_apogee": 150000})
    monkeypatch.setattr("sys.argv", ["main.py", "candidate.yaml"])
    monkeypatch.setattr(main, "load_config", lambda path: {})
    monkeypatch.setattr(main, "simulate", lambda *args, **kwargs: result)
    monkeypatch.setattr(main, "write_history", lambda *args: pytest.fail("Rejected run wrote history"))
    with pytest.raises(SystemExit) as caught:
        main.main()
    message = str(caught.value)
    assert termination in message
    assert "candidate.yaml" in message
    assert "pump.oxidizer_pump.max_power: margin=-0.1569 kW" in message
    assert "goal_apogee" not in message  # Unassessed is not a measured violation.


def test_cli_keeps_unexplained_empty_history_as_an_error(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "candidate.yaml"])
    monkeypatch.setattr(main, "load_config", lambda path: {})
    monkeypatch.setattr(main, "simulate", lambda *args, **kwargs: SimResult())
    with pytest.raises(RuntimeError, match="no time steps.*time_limit"):
        main.main()


def test_structural_load_csv_preserves_times_stations_signs_and_per_metre_units(tmp_path):
    history = []
    for time, scale in ((.5, 1), (1.25, 2)):
        history.append({
            "kinematics": SimpleNamespace(t=time),
            "mass_properties": {"cell_widths": [.1, .25]},
            "loads": {"station": [.05, .225], "axial": [1000 * scale, -2000 * scale],
                      "axial_aero": [100 * scale, 500 * scale],
                      "normal": [-200 * scale, 1000 * scale],
                      "shear": [-200 * scale, 800 * scale],
                      "bending": [-20 * scale, 180 * scale]},
        })
    path = tmp_path / "loads" / "flight_structural_loads.csv"
    main.write_structural_loads(history, path)
    with path.open(newline="") as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    assert [(r["time_s"], r["station_m"]) for r in rows] == [(.5, .05), (.5, .225), (1.25, .05), (1.25, .225)]
    assert rows[0]["internal_axial_force_kN"] == 1
    assert rows[1]["internal_axial_force_kN"] == -2
    assert rows[0]["distributed_normal_force_kN_m"] == -2
    assert rows[1]["distributed_normal_force_kN_m"] == 4
    assert rows[1]["distributed_axial_aero_force_kN_m"] == 2
    assert rows[3]["internal_shear_force_kN"] == 1.6
    assert rows[3]["internal_bending_moment_kN_m"] == .36
    # Integrating the exported density recovers the original distributed load.
    assert sum(r["distributed_normal_force_kN_m"] * r["cell_width_m"] for r in rows[:2]) == pytest.approx(.8)


def test_structural_load_csv_keeps_every_tenth_time_and_final_time(tmp_path):
    history = [{"kinematics": SimpleNamespace(t=i * .1),
                "mass_properties": {"cell_widths": [.1, .1]},
                "loads": {"station": [.05, .15], "axial": [0, 0],
                          "axial_aero": [0, 0], "normal": [0, 0],
                          "shear": [0, 0], "bending": [0, 0]}}
               for i in range(24)]
    path = tmp_path / "loads.csv"
    main.write_structural_loads(history, path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert [float(r["time_s"]) for r in rows[::2]] == pytest.approx([0., 1., 2., 2.3])
    assert [float(r["station_m"]) for r in rows] == [.05, .15] * 4
