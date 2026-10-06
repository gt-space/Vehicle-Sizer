import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from flight_plots import _valve_history, plot_valve_actuations
from main import write_events


class ValveHistoryTests(unittest.TestCase):
    def history(self):
        events = tuple(dict(time_s=t, kind="branch", component="valve", event="switch",
                            count=i, was_open=before, is_open=after)
                       for i, (t, before, after) in enumerate(
                           [(0.2, False, True), (0.3, True, False)], 1))
        fluid = SimpleNamespace(branch={"valve": {"is_open": False}}, events=events,
                                event_counts={"branch:valve:switch": 2})
        return [{"kinematics": SimpleNamespace(t=1.0, dt=1.0),
                 "plant": SimpleNamespace(fluids=fluid)}]

    def test_multiple_switches_between_identical_endpoint_states(self):
        history = self.history()
        self.assertEqual(_valve_history(history, "valve"),
                         ([0.0, 0.2, 0.3, 1.0], [False, True, False, False], [0, 1, 2, 2]))
        with TemporaryDirectory() as directory:
            path = Path(directory)
            write_events(history, path / "events.csv")
            with (path / "events.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["time_s"] for row in rows], ["0.2", "0.3"])
            plot_valve_actuations(history, path / "valves.png", burnout=0.8)
            self.assertTrue((path / "valves.png").is_file())

    def test_no_switches(self):
        history = self.history()
        history[0]["plant"].fluids.events = ()
        history[0]["plant"].fluids.event_counts = {}
        self.assertEqual(_valve_history(history, "valve"), ([1.0], [False], [0]))

    def test_switch_state_history(self):
        history = self.history()
        fluid = history[0]["plant"].fluids
        fluid.branch = {"valve": {"is_switched": True}}
        fluid.events = (dict(time_s=.2, kind="branch", component="valve", event="switch",
                             count=1, was_switched=False, is_switched=True),)
        fluid.event_counts = {"branch:valve:switch": 1}
        self.assertEqual(_valve_history(history, "valve"),
                         ([0., .2, 1.], [False, True, True], [0, 1, 1]))
        with TemporaryDirectory() as directory:
            plot_valve_actuations(history, Path(directory) / "switch.png")

    def test_event_csv_preserves_new_component_transition_fields(self):
        history = self.history()
        history[0]["plant"].fluids.events = (
            dict(time_s=.2, kind="branch", component="switch", event="switch",
                 was_switched=False, is_switched=True, effective_CdA=1.2e-5),)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            write_events(history, path)
            with path.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['is_switched'], 'True')
            self.assertEqual(float(row['effective_CdA']), 1.2e-5)


if __name__ == "__main__":
    unittest.main()
