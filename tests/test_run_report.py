from types import SimpleNamespace

import numpy as np

from run_report import collect_design_summary, build_run_tables
from simulation_types import SimResult


def fixture():
    cfg = {
        "vehicle": {"OMLD": .3, "sections": [
            {"type": "inter_tank", "name": "Payload"},
            {"type": "prop_tank", "tank_id": "custom_oxidizer"},
            {"type": "fin_can"},
        ]},
        "tanks": {"custom_oxidizer": {"type": "propellant", "material": "aluminum_6061"}},
        "engine": {"exit_pressure": 70000., "cf_efficiency": .95, "cstar_efficiency": .85},
        "prop_system": {"thrust_target": 16000., "MR_target": 2.1},
    }
    sections = [SimpleNamespace(start_station=x, length=1., mass=np.array([m]),
                                volume=.2, OMLD=.3, wall_thickness=.0014)
                for x, m in ((0., 2.), (1., 3.), (2., 4.))]
    vehicle = SimpleNamespace(sections=sections, engine_start_station=2.,
                              engine=SimpleNamespace(length=.5, mass=20.), length=3., Ixx=1., Iyy=20.)
    propulsion = SimpleNamespace(
        initial_states={"custom_oxidizer": {"T": 95.}}, Pc_target=2e6, MR_target=2.1,
        expansion_ratio=5., cstar_efficiency=.85, cf_efficiency=.95, throat_area=.01, mdot_total=8.,
        combustion_properties=SimpleNamespace(evaluate=lambda **kwargs: SimpleNamespace(Cf=1.5)),
    )
    design = collect_design_summary(cfg, vehicle, propulsion)
    node = {"custom_tank_node": {"tank_id": "custom_oxidizer", "m_liq": 10., "mass": 11., "P": 2e6, "T_liq": 95.},
            "custom_pump_out": {"P": 3e6}}
    def state(t, mass, mode, thrust, speed):
        return dict(kinematics=SimpleNamespace(t=t, vx=0., vz=speed),
                    atmosphere=SimpleNamespace(Ma=speed / 340),
                    mass_properties={"total_mass": mass, "Ixx": mass / 10, "Iyy": mass * 2},
                    plant=SimpleNamespace(fluids=SimpleNamespace(node=node,
                        propulsion=SimpleNamespace(mode=mode, thrust=thrust)),
                        aero=SimpleNamespace(A=0., N=0.)),
                    forces={"thrust": thrust, "twr": 3.64}, engine_on=mode == "combusting")
    initial = state(0, 40, "combusting", 16000, 0)
    result = SimResult(design_summary=design, initial_state=initial, initial_mass=40, dry_mass=29,
                       history=[state(1, 38, "combusting", 17000, 100), state(2, 30, "shutdown", 0, 200)],
                       rail_exit_time=1., burn_complete=True, burn_duration=2., apogee=1000., max_q=10000.)
    return cfg, result


def test_sections_are_dynamic_and_engine_not_double_counted():
    cfg, result = fixture()
    sections = result.design_summary["sections"]
    assert [s[0] for s in sections] == ["Payload", "custom_oxidizer", "Fin can / boattail", "Engine"]
    assert sum(s[3] for s in sections) == result.dry_mass
    output = "\n".join(build_run_tables(cfg, result))
    assert "Avionics" not in output
    assert "custom_pump_out" in output
    assert "custom_oxidizer" in output
    assert "Izz (assumed = Iyy)" in output
    assert "Initial pressurant / other stored fluids" in output


def test_report_uses_launch_snapshot_not_first_endpoint():
    cfg, result = fixture()
    tables = build_run_tables(cfg, result)
    mass_rows = [line for line in tables[3].splitlines() if "Launch mass" in line]
    assert "40.00" in mass_rows[0]
    assert "10.000" in tables[2]  # Initial liquid load.
    assert "Shutdown" in tables[-1] and "60.000" in tables[-1]


def test_partial_run_does_not_claim_shutdown_or_apogee():
    cfg, result = fixture()
    result.history = result.history[:1]
    result.apogee = None
    result.burn_complete = False
    tables = build_run_tables(cfg, result)
    assert "N/A" in next(line for line in tables[3].splitlines() if "Mass after shutdown" in line)
    assert "Not reached" in tables[5]
    assert "(incomplete)" in tables[6]


def test_proper_acceleration_is_not_vertical_coordinate_acceleration():
    cfg, result = fixture()
    for state in [result.initial_state, *result.history]:
        state["forces"]["thrust"] = 0
        state["plant"].aero.N = 0
        state["plant"].aero.A = 0
    table = build_run_tables(cfg, result)[5]
    assert "0.000 g" in table


def test_main_prints_tables(monkeypatch, capsys, tmp_path):
    import sys
    import main
    cfg, result = fixture()
    cfg["simulation"] = {"output": str(tmp_path / "history.csv"), "plot": str(tmp_path / "flight.png")}
    monkeypatch.setattr(sys, "argv", ["main.py", "custom.yaml"])
    monkeypatch.setattr(main, "load_config", lambda path: cfg)
    monkeypatch.setattr(main, "simulate", lambda *args, **kwargs: result)
    monkeypatch.setattr(main, "history_rows", lambda history: [])
    monkeypatch.setattr(main, "write_history", lambda *args: None)
    monkeypatch.setattr(main, "write_events", lambda *args: None)
    monkeypatch.setattr(main, "write_structural_loads", lambda *args: None)
    monkeypatch.setitem(sys.modules, "flight_plots", SimpleNamespace(plot_flight=lambda *args: {}))
    main.main()
    output = capsys.readouterr().out
    for label in ("Vehicle sections", "Tanks", "Vehicle mass", "Pressure ladder",
                  "Kinematics", "Engine", "Body-axis inertia", "Payload", "custom_pump_out"):
        assert label in output
    assert "Avionics" not in output
