import pytest

import main
from constraints import finalize
from simulation_types import SimResult


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
