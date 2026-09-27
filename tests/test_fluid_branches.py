"""Branch physics and event boundaries, independent of any IDA installation."""
from copy import deepcopy

import numpy as np
import pytest

from Fluids import FluidBranch as dae
from Fluids.FluidState import FluidState


def definition(**extra):
    return {"from": "up", "to": "down", "fluid": "declared_fluid",
            "CdA": 1e-4, "design_mdot": 0.5, **extra}


def fluid(phase):
    return FluidState("water" if phase == "liquid" else "Nitrogen", phase,
                      {"rho": 1000.0 if phase == "liquid" else 5.0,
                       "h": 42000.0, "T": 300.0, "R": 296.8, "gamma": 1.4})


def nodes(up=300000.0, down=100000.0):
    return {"up": {"P": up, "cstar": 1500.0}, "down": {"P": down}}


@pytest.mark.parametrize("kind,phase,extra", [
    ("LossComponent", "liquid", {}),
    ("LossComponent", "gas", {}),
    ("PumpComponent", "liquid", {"dP": 2e5, "gas_CdA": 2e-4}),
    ("PumpComponent", "gas", {"dP": 2e5, "gas_CdA": 2e-4}),
    ("RegulatorComponent", "gas", {"target_pressure": 2e5}),
    ("BangBangValveComponent", "gas", {"target_pressure": 2e5,
                                        "pressure_band": 1e4, "initially_open": True}),
    ("NozzleComponent", "gas", {"At": 1e-3, "Cd": 0.9}),
])
@pytest.mark.parametrize("up,down", [(3e5, 1e5), (3e5, 2.8e5), (1e5, 3e5), (2e5, 2e5)])
@pytest.mark.parametrize("port_offset", [0, 2.5e5])
def test_trial_order_does_not_change_component_state(kind, phase, extra, up, down, port_offset):
    options = {"phase": phase} if kind in {"LossComponent", "PumpComponent"} else {}
    branch = getattr(dae, kind)("test", definition(**extra), **options)
    source = {"source": fluid(phase)}
    pressure = nodes(up, down)
    pressure["down"]["port_pressure"] = {"declared_fluid": down + port_offset}
    before = deepcopy(vars(branch))
    arguments = deepcopy((source, pressure))

    trial = branch.evaluate({"mdot": 0.2}, pressure, source, {"main": 1})
    assert np.all(np.isfinite(branch.residual(trial, pressure)))
    assert trial.mdot == .2
    assert trial.trial_values == {"mdot": 0.2}
    branch.evaluate({"mdot": 4.0}, nodes(down=240000), source, {"main": 1})
    repeated = branch.evaluate({"mdot": 0.2}, pressure, source, {"main": 1})
    assert repeated == trial
    assert vars(branch) == before
    assert (source, pressure) == arguments


def test_liquid_restriction_solves_the_known_forward_and_reverse_flows():
    branch = dae.LossComponent("loss", definition())
    # CdA*sqrt(2*rho*abs(dP)) = 2 kg/s for the selected values.
    for up, down, mdot in [(3e5, 1e5, 2.0), (1e5, 3e5, -2.0)]:
        pressure = nodes(up, down)
        state = branch.evaluate({"mdot": mdot}, pressure, {"water": fluid("liquid")},
                                {"main": 1 if mdot > 0 else -1})
        np.testing.assert_allclose(branch.residual(state, pressure), 0, atol=1e-14)


@pytest.mark.parametrize("component", [dae.PumpComponent, dae.LossComponent])
def test_phase_changes_are_explicit_and_invalid_trials_do_not_switch(component):
    branch = component("path", definition(dP=2e5, gas_CdA=2e-4))
    before = deepcopy(vars(branch))
    with pytest.raises(ValueError, match="between solves"):
        branch.evaluate({"mdot": 1}, nodes(), {"gas": fluid("gas")}, {"main": 1})
    assert vars(branch) == before
    assert branch.set_phase("gas")
    assert not branch.set_phase("gas")
    state = branch.evaluate({"mdot": 1}, nodes(), {"gas": fluid("gas")}, {"main": 1})
    assert np.all(np.isfinite(branch.residual(state, nodes())))
    if component is dae.PumpComponent:
        assert state.properties["pump_active"] is False
        assert branch.gas_cda == 2e-4
    with pytest.raises(ValueError, match="Unsupported"):
        branch.set_phase("unknown")


def test_branch_owns_parameters_and_modes_but_no_committed_flow_state():
    config = definition(nested={"value": 1})
    branch = dae.LossComponent("loss", config)
    config["nested"]["value"] = 9
    assert branch.parameters["nested"]["value"] == 1
    before = deepcopy(vars(branch))
    source = {"water": fluid("liquid")}
    trial = branch.evaluate({"mdot": 20}, nodes(), source, {"main": 1})
    trial.flows["main"]["mdot"] = 99
    source["water"].properties["h"] = -1
    assert vars(branch) == before
    assert not hasattr(branch, "state") and not hasattr(branch, "commit")


def test_closed_valve_has_a_zero_flow_equation_without_donor_properties():
    valve = dae.BangBangValveComponent("valve", definition(
        target_pressure=2e5, pressure_band=1e4, initially_open=False))
    layout = valve.variable_names
    for q in (0.0, 0.4):
        state = valve.evaluate({"mdot": q}, {}, {}, {})
        np.testing.assert_allclose(valve.residual(state, {}), [q])
        assert not state.enabled and state.mdot == 0
    assert valve.event_values(nodes(down=190000))["switch"] == 0
    assert not valve.is_open  # Evaluating the root must not toggle the valve.
    assert valve.set_open(True)
    assert valve.variable_names == layout
    assert valve.event_values(nodes(down=190000))["switch"] > 0
    assert valve.event_values(nodes(down=210000))["switch"] == 0
    assert valve.set_open(False)
    assert valve.event_values(nodes(down=210000))["switch"] > 0
    valve.set_enabled(False)
    assert valve.event_values(nodes()) == {}


@pytest.mark.parametrize("down,flow_case", [(250000, "closed"), (150000, "full"), (200000, "interior")])
def test_regulator_limit_roots(down, flow_case):
    branch = dae.RegulatorComponent("reg", definition(target_pressure=2e5))
    pressure = nodes(down=down)
    source = {"gas": fluid("gas")}
    sample = branch.evaluate({"mdot": 0}, pressure, source, {"main": 1})
    capacity = sample.properties["max_mdot"]
    q = 0 if flow_case == "closed" else capacity * (0.5 if flow_case == "interior" else 1)
    state = branch.evaluate({"mdot": q}, pressure, source, {"main": 1})
    np.testing.assert_allclose(branch.residual(state, pressure), 0, atol=1e-14)
    assert 0 <= state.properties["opening_fraction"] <= 1


def test_disabled_pump_retains_zero_flow_equation():
    pump = dae.PumpComponent("pump", definition(dP=2e5, gas_CdA=2e-4))
    pump.set_enabled(False)
    assert pump.variable_names == ("mdot",)
    trial = pump.evaluate({"mdot": 0.3}, {}, {}, {})
    assert not trial.properties["pump_active"]
    np.testing.assert_allclose(pump.residual(trial, {}), [0.3])


@pytest.mark.parametrize("up,down", [(3e5, 1e5), (3e5, 2.8e5), (1e5, 3e5), (2e5, 2e5)])
@pytest.mark.parametrize("port_offset", [0, 2.5e5])
def test_shutdown_nozzle_changes_layout_only_on_explicit_transition(up, down, port_offset):
    config = definition(At=1e-3, Cd=0.9)
    branch = dae.NozzleComponent("nozzle", config)
    pressure = nodes(up, down)
    pressure["down"]["port_pressure"] = {"declared_fluid": down + port_offset}
    assert branch.variable_names == ("mdot",)
    sources = {phase: fluid(phase) for phase in ("gas", "liquid")}
    branch.set_mode(False, ("gas", "liquid"))
    raw = {"mdot_gas": 0.3, "mdot_liquid": 0.3, "gas_area_fraction": 0.4}
    assert set(raw) == {"mdot_gas", "mdot_liquid", "gas_area_fraction"}
    directions = {"gas": 1, "liquid": 1}
    before = deepcopy(vars(branch))
    trial = branch.evaluate(raw, pressure, sources, directions)
    assert np.all(np.isfinite(branch.residual(trial, pressure)))
    assert trial.mdot == pytest.approx(.6)
    assert trial.dP == up - down - port_offset
    assert vars(branch) == before
    with pytest.raises(ValueError, match="between solves"):
        branch.evaluate(raw, pressure, {"gas": sources["gas"]}, directions)
    assert vars(branch) == before
    # Equal pressure has zero capacity; trial evaluation must remain finite.
    equal = nodes(1e5, 1e5)
    trial = branch.evaluate(raw, equal, sources, directions)
    np.testing.assert_allclose(branch.residual(trial, equal), [0.3, 0.3])
    assert trial.properties["thrust"] == 0
    branch.set_mode(False, ())
    state = branch.evaluate({}, pressure, {}, {})
    assert branch.variable_names == () and branch.residual(state, pressure).size == 0
    assert state.properties["thrust"] == 0


@pytest.mark.parametrize("phases", [("gas",), ("liquid",), ("gas", "liquid")])
@pytest.mark.parametrize("q", [0.0, 0.7, -0.3])
def test_shutdown_constituent_mixing_preserves_original_equations(phases, q):
    config = definition(At=1e-3, Cd=0.9, design_mdot=2.5)
    branch = dae.NozzleComponent("nozzle", config)
    sources = {f"{phase}_{i}": fluid(phase) for phase in phases for i in range(2)}
    # Different densities and gas constants exercise per-phase constituent mixing.
    for i, source in enumerate(sources.values()):
        source.properties["rho"] *= 1 + i
        source.properties["R"] *= 1 + i * 0.1
    raw = {f"mdot_{phase}": q for phase in phases}
    if len(phases) == 2:
        raw["gas_area_fraction"] = 0.4
    directions = {phase: -1 if q < 0 else 1 for phase in phases}
    pressure = nodes()
    branch.set_mode(False, phases)
    trial = branch.evaluate(raw, pressure, sources, directions)
    assert np.all(np.isfinite(branch.residual(trial, pressure)))
    assert trial.mdot == pytest.approx(len(phases)*q)


@pytest.mark.parametrize("q", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("loss_cda", [None, 1e-4])
def test_pump_curve_and_signed_losses_follow_momentum_balance(q, loss_cda):
    config = definition(dP=2e5, gas_CdA=2e-4, CdA=loss_cda,
                        head_model=lambda mdot: 2e5 - 3e4 * mdot ** 2)
    branch = dae.PumpComponent("pump", config)
    pressure, source = nodes(up=1e5, down=3e5), {"liquid": fluid("liquid")}
    args = ({"mdot": q}, pressure, source, {"main": -1 if q < 0 else 1})
    trial = branch.evaluate(*args)
    head = 2e5 - 3e4*q*q
    loss = 0. if loss_cda is None else np.sign(q)*(q/loss_cda)**2/(2*1000.)
    np.testing.assert_allclose(branch.residual(trial, pressure), [(-2e5 + head - loss)/max(abs(head), 1e5)])



@pytest.mark.parametrize("phase,key,value", [
    ("gas", "T", 0.), ("gas", "T", np.nan),
    ("gas", "R", 0.), ("gas", "R", np.inf),
    ("gas", "gamma", 1.), ("gas", "gamma", np.nan),
    ("liquid", "rho", 0.), ("liquid", "rho", np.nan),
])
def test_invalid_flow_properties_are_trial_domain_errors(phase, key, value):
    from Fluids import TrialDomainError
    branch = dae.LossComponent("loss", definition(), phase=phase)
    source = fluid(phase)
    source.properties[key] = value
    with pytest.raises(TrialDomainError):
        trial = branch.evaluate({"mdot": 0.1}, nodes(), {"source": source}, {"main": 1})
        branch.residual(trial, nodes())


def test_branch_distinguishes_trial_errors_from_config_layout_and_mode_errors():
    from Fluids import TrialDomainError
    branch = dae.LossComponent("loss", definition())
    with pytest.raises(TrialDomainError, match="non-finite trial"):
        branch.evaluate({"mdot": np.nan}, nodes(), {"water": fluid("liquid")}, {"main": 1})
    with pytest.raises(TrialDomainError, match="pressures"):
        branch.evaluate({"mdot": 0.1}, nodes(up=np.inf), {"water": fluid("liquid")}, {"main": 1})
    for action in (
        lambda: dae.LossComponent("bad", definition(CdA=-1)),
        lambda: branch.evaluate({"wrong": 0.1}, nodes(), {}, {}),
        lambda: branch.evaluate({"mdot": 0.1}, nodes(), {"gas": fluid("gas")}, {"main": 1}),
    ):
        with pytest.raises(ValueError) as error:
            action()
        assert type(error.value) is ValueError


def test_pump_curve_domain_error_is_distinct_from_callback_programming_error():
    from Fluids import TrialDomainError
    branch = dae.PumpComponent("pump", definition(dP=2e5, gas_CdA=2e-4,
                                                  head_model=lambda q: np.nan))
    trial = branch.evaluate({"mdot": 0.1}, nodes(), {"water": fluid("liquid")}, {"main": 1})
    with pytest.raises(TrialDomainError, match="finite head"):
        branch.residual(trial, nodes())
    def broken_callback(q):
        raise ValueError("head curve implementation error")
    branch.head_model = broken_callback
    with pytest.raises(ValueError, match="implementation error") as error:
        branch.residual(trial, nodes())
    assert type(error.value) is ValueError
