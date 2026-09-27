"""Record ownership and reporting, without a solver or property provider."""

from copy import deepcopy

import numpy as np
import pytest

from Fluids.FluidState import BranchState, FluidState, NodeState


def test_node_trial_snapshot_and_report_have_separate_ownership():
    raw = {"m": 2.0, "U": 1000.0, "P": 2e5, "T": 300.0}
    fluid = FluidState("Nitrogen", "gas", {"h": 500.0})
    node = NodeState(
        trial_values=raw, fluids={"Nitrogen": fluid},
        properties={"port_pressure": {"gas": 2e5}},
        evaluation_data={"closures": np.array([0., 0.])},
    )
    branch = BranchState(trial_values={"mdot": 0.2}, flows={
        "main": {"mdot": 0.2, "direction": 1, "fluid": fluid},
    })
    assert branch.flows["main"]["fluid"] is node.fluids["Nitrogen"]
    raw["m"] = 99.0
    assert node["m"] == 2.0
    assert node.get("P") == 2e5
    assert "closures" not in node
    with pytest.raises(KeyError):
        node["closures"]

    # Accepted snapshots preserve internal data and detach shared donor records.
    accepted_node, accepted_branch = deepcopy((node, branch))
    fluid.properties["h"] = -1.0
    node.properties["port_pressure"]["gas"] = 0.0
    node.evaluation_data["closures"][0] = 10.0
    branch.flows["main"]["mdot"] = 99.0
    assert accepted_node.fluids["Nitrogen"]["h"] == 500.0
    assert accepted_branch.flows["main"]["fluid"] is accepted_node.fluids["Nitrogen"]
    assert accepted_branch.mdot == 0.2
    np.testing.assert_array_equal(accepted_node.evaluation_data["closures"], [0., 0.])

    report = accepted_node.as_dict()
    assert not {"evaluation_data", "closures"} & report.keys()
    report["fluids"]["Nitrogen"]["h"] = 0.0
    report["port_pressure"]["gas"] = 0.0
    assert accepted_node["port_pressure"]["gas"] == 2e5
    assert accepted_node.fluids["Nitrogen"]["h"] == 500.0


def test_boundary_and_fluid_report_imports_are_detached_data_not_solver_states():
    data = {"P": 1e5, "fluids": {"Nitrogen": {"phase": "gas", "h": 500.0}}}
    node = NodeState.from_dict(data)
    assert node.trial_values == {} and node.evaluation_data == {}
    assert node.properties == {"P": 1e5}
    assert node.as_dict() == data
    data["fluids"]["Nitrogen"]["h"] = 0.0
    assert node.fluids["Nitrogen"]["h"] == 500.0
    fluid = node.fluids["Nitrogen"]
    assert FluidState.from_dict("Nitrogen", fluid.as_dict()) == fluid
    assert dict(fluid) == {"fluid": "Nitrogen", "phase": "gas", "h": 500.0}
    assert len(fluid) == len(dict(fluid))
    with pytest.raises(KeyError):
        fluid["rho"]  # Missing properties are not fabricated by the record.
    with pytest.raises(TypeError):
        fluid["h"] = 0.0  # Builders must explicitly write properties.


def test_branch_signed_phase_flows_and_reports_preserve_donor_identity():
    gas = FluidState("Nitrogen", "gas", {"h": 500.0})
    liquid = FluidState("Oxygen", "liquid", {"h": 100.0})
    flows = {
        "gas": {"mdot": -0.2, "direction": -1, "fluid": gas},
        "liquid": {"mdot": 0.7, "direction": 1, "fluid": liquid},
    }
    branch = BranchState(trial_values={"mdot_gas": -0.2, "mdot_liquid": 0.7},
                         flows=flows, dP=1e5, properties={"thrust": 10.0})
    flows["gas"]["mdot"] = 99.0
    assert branch.mdot == pytest.approx(0.5)
    assert branch.phase_mdot("gas") == -0.2
    assert branch.phase_mdot("liquid") == 0.7
    assert branch["mdot_gas"] == -0.2 and branch["thrust"] == 10.0
    assert len(branch) == len(dict(branch))
    report = branch.as_dict()
    assert report["flows"]["gas"]["fluid_name"] == "Nitrogen"
    assert report["flows"]["gas"]["fluid"]["phase"] == "gas"
    report["flows"]["gas"]["fluid"]["h"] = 0.0
    report["flows"]["gas"]["mdot"] = 0.0
    assert gas["h"] == 500.0 and branch.phase_mdot("gas") == -0.2


def test_closed_branch_keeps_trial_unknown_but_transports_nothing():
    from Fluids.FluidBranch import LossComponent
    from Fluids.FluidNode import fluxes

    branch = LossComponent("valve", {"from": "tank", "to": "chamber", "CdA": 1e-4})
    branch.set_enabled(False)
    raw = {"mdot": 0.2}
    state = branch.evaluate(raw, {}, {}, {})
    raw["mdot"] = 99.0
    assert state.trial_values == {"mdot": 0.2}
    assert state["mdot"] == 0.2  # Mapping reads the unknown, not transported flow.
    assert state.mdot == state.phase_mdot("gas") == state.phase_mdot("liquid") == 0
    assert state.flows == {} and fluxes([(1, state)]) == (0.0, 0.0)
    np.testing.assert_allclose(branch.residual(state, {}), [0.2])


@pytest.mark.parametrize("record, kwargs", [
    (FluidState, {"fluid": "Nitrogen", "phase": "gas", "properties": {"phase": "liquid"}}),
    (NodeState, {"trial_values": {"P": 1e5}, "properties": {"P": 2e5}}),
    (NodeState, {"properties": {"fluids": {}}}),
    (BranchState, {"trial_values": {"mdot": 1.0}, "properties": {"mdot": 2.0}}),
    (BranchState, {"properties": {"enabled": False}}),
])
def test_record_construction_rejects_ambiguous_names(record, kwargs):
    with pytest.raises(ValueError):
        record(**kwargs)
