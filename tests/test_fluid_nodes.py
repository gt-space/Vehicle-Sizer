"""Continuous node physics, thermodynamic derivatives, and explicit mode boundaries."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import root

from FluidTables.LookupTables import LookupTable
from FluidTables.PropertyModels import (
    CoolPropPropertySource, PureFluidProperties, SaturationProperties,
    TablePureFluidPropertySource, CombustionProperties,
)
from Fluids.FluidNode import (
    BoundaryComponent, JunctionComponent, VolumeComponent,
    PropellantTankComponent, CombustorComponent,
)
from Fluids import TrialDomainError
from Fluids.FluidState import BranchState, FluidState


@dataclass
class Geometry:
    volume: float = 1.0

    def fill_state(self, liquid_volume):
        return {"fill_height": liquid_volume, "liquid_contact_area": liquid_volume,
                "ullage_contact_area": self.volume - liquid_volume}

    def axial_mass(self, mass=None, *, liquid_volume=None, liquid_mass=None, ullage_mass=None):
        return [mass] if mass is not None else [liquid_mass, ullage_mass]


class AnalyticProperties:
    """Compressible/thermally expanding liquid plus ideal gas, with exact slopes."""
    def state_pt(self, fluid, pressure, temperature):
        liquid = fluid == "water"
        rho = 1000 + 1e-6 * pressure - 0.2 * (temperature - 300) if liquid else pressure / (300 * temperature)
        u = 2000 * temperature + 1e-4 * pressure if liquid else 700 * temperature
        return PureFluidProperties(pressure, temperature, rho, u + pressure / rho, u,
                                   300., 1.4, 1e-5, 0.03, 1000., 1 / temperature)

    def derivatives_pt(self, fluid, pressure, temperature):
        if fluid == "water":
            return dict(drho_dP=1e-6, drho_dT=-0.2, du_dP=1e-4, du_dT=2000.)
        return dict(drho_dP=1 / (300 * temperature), drho_dT=-pressure / (300 * temperature**2),
                    du_dP=0., du_dT=700.)

    def supports_saturation(self, fluid):
        return False

    def state_bounds(self, fluid):
        return (1., 1e8), (1., 5000.)


def wet_tank():
    props = AnalyticProperties()
    pressure, tl, tg = 2e5, 290., 310.
    liquid, gas = props.state_pt("water", pressure, tl), props.state_pt("Nitrogen", pressure, tg)
    ml, mg = 0.6 * liquid.rho, 0.4 * gas.rho
    node = PropellantTankComponent("tank", {
        "geometry": Geometry(), "liquid_fluid": "water", "gas_fluid": "Nitrogen",
        "state0": {"P": pressure, "T": tl, "gas_T": tg,
                   "m_liq": ml, "m_ull": mg, "U_liq": ml * liquid.u, "U_ull": mg * gas.u}},
        fluid_properties=props)
    return node, node.initial_values()


def branch(fluid, mdot, direction=1, enabled=True):
    return BranchState(enabled=enabled, flows={"main": {"fluid": fluid, "mdot": mdot, "direction": direction}})


def test_shared_keyword_contract_for_node_and_branch_trial_evaluation():
    from Fluids.FluidBranch import LossComponent

    tank, trial_values = wet_tank()
    ambient = BoundaryComponent("ambient", {"P": 1e5})
    node_states_by_id = {"ambient": ambient.evaluate(
        trial_values={}, boundary_values={"P": 1.5e5})}
    node_state = tank.evaluate(trial_values=trial_values,
                               node_states_by_id=node_states_by_id)
    node_states_by_id["tank"] = node_state
    assert set(trial_values) == set(tank.variable_names)
    np.testing.assert_allclose(tank.residual(
        node_state=node_state,
        trial_derivatives={key: 0.0 for key in tank.differential_variable_names}),
        0.0, atol=1e-8)

    loss = LossComponent("outlet", {"from": "tank", "to": "ambient",
                                    "from_port": "liquid", "CdA": 1e-4})
    assert loss.variable_names == ("mdot",)
    assert loss.differential_variable_names == ()
    branch_state = loss.evaluate(
        trial_values={"mdot": 0.1}, node_states_by_id=node_states_by_id,
        source_fluids=tank.outlet(node_state=node_state, port="liquid"),
        directions={"main": 1})
    assert np.isfinite(loss.residual(
        branch_state=branch_state, node_states_by_id=node_states_by_id)).all()
    for component in (tank, loss):
        with pytest.raises(ValueError, match="requires variables"):
            component._check_trial_values({"wrong_name": 1.0})


def test_junction_evaluates_mixed_state_and_detaches_output():
    boundary = BoundaryComponent("supply", {"P": 2e5, "fluids": {
        "Nitrogen": {"phase": "gas", "h": 300000., "T": 300.}}})
    supply = boundary.evaluate({}, boundary_values={"P": 3e5})
    assert supply["P"] == 3e5 and boundary.definition["P"] == 2e5
    junction = JunctionComponent("junction", {'fluid':'Nitrogen', 'phase':'gas',
                                 'state0':{'P':1e5,'T':300.}}, fluid_properties=AnalyticProperties())
    # Reverse flow in a branch nominally leaving the junction is an inlet.
    adjacent = [(-1, branch(supply.fluids["Nitrogen"], -2., -1)),
                (-1, branch(supply.fluids["Nitrogen"], 2.)),
                (1, branch(supply.fluids["Nitrogen"], 100., enabled=False))]
    trial = junction.evaluate({"P": 1.5e5, 'T':300.}, adjacent)
    np.testing.assert_allclose(junction.residual(trial, {}, adjacent), 0)
    assert junction.differential_variable_names == ()
    assert list(junction.outlet(trial)) == ["Nitrogen"]
    assert trial.fluids["Nitrogen"] is not supply.fluids["Nitrogen"]
    assert trial.fluids['Nitrogen']['P'] == 1.5e5
    output = junction.output_state(trial)
    output.fluids["Nitrogen"].properties["h"] = -999
    assert supply.fluids["Nitrogen"].properties["h"] == 300000.
    assert boundary.residual(supply, {}).size == 0


def test_gas_volume_has_analytic_blowdown_rates_and_fixed_inventory_closures():
    props = AnalyticProperties()
    gas = props.state_pt("Nitrogen", 2e5, 300.)
    node = VolumeComponent("copv", {"fluid": "Nitrogen", "geometry": Geometry(),
        "state0": {"P": gas.P, "T": gas.T, "m": gas.rho, "U": gas.rho * gas.u}},
        fluid_properties=props)
    values = node.initial_values()
    trial = node.evaluate(values)
    q, heat = 0.1, 25.
    adjacent = [(-1, branch(node.outlet(trial)["Nitrogen"], q))]
    rates = {"m": -q, "U": -q * gas.h + heat}
    np.testing.assert_allclose(node.residual(trial, rates, adjacent, heat_rate={"gas": heat}), 0, atol=1e-10)
    changed = node.evaluate({**values, "U": values["U"] + 100})
    assert changed["U"] == values["U"] + 100  # Never overwrite trial inventory with EOS energy.
    assert node.residual(changed, rates, adjacent, heat_rate={"gas": heat})[-1] == pytest.approx(100)
    output = node.output_state(trial)
    assert output["mass"] == values["m"] and not output.evaluation_data
    output.fluids["Nitrogen"].properties["h"] = -1
    assert trial.fluids["Nitrogen"].properties["h"] == gas.h


def test_wet_tank_preserves_trial_inventory_and_resolves_port_pressure():
    node, values = wet_tank()
    before = deepcopy(node.definition)
    trial = node.evaluate(values, axial_specific_force=20.)
    for name in ("P", "U_liq", "U_ull"):
        assert trial[name] == values[name]
    assert node.outlet(trial, port="liquid")["water"].phase == "liquid"
    assert node.outlet(trial, port="ullage")["Nitrogen"].phase == "gas"
    assert trial["port_pressure"]["liquid"] > trial["P"]
    with pytest.raises(ValueError, match="unambiguous"):
        node.outlet(trial)
    np.testing.assert_allclose(trial.evaluation_data["closures"], 0, atol=1e-10)
    node.evaluate({**values, "P": 2.1e5})
    repeated = node.evaluate(values, axial_specific_force=20.)
    np.testing.assert_allclose(repeated.evaluation_data["volume_gradient"], trial.evaluation_data["volume_gradient"])
    assert node.definition == before and not hasattr(node, "state")
    output = node.output_state(trial)
    assert sum(output["axial_mass"]) == pytest.approx(values["m_liq"] + values["m_ull"])


def test_liquid_volume_gradient_matches_perturbed_thermodynamic_solutions():
    node, values = wet_tank()
    trial = node.evaluate(values)
    x = np.array([values[k] for k in node.differential_variable_names])
    z = np.array([values[k] for k in ("P", "T_liq", "T_ull")])
    zscale = np.array([2e5, 300., 300.])
    escale = np.array([values["U_liq"], values["U_ull"], 1.])
    direction = np.array([-1., 0.02, -6e5, 8000.])
    def volume_at(inventory):
        def residual(guess):
            vals = dict(zip(node.differential_variable_names, inventory))
            vals.update(zip(("P", "T_liq", "T_ull"), guess * zscale))
            return np.array(node.evaluate(vals).evaluation_data["closures"]) / escale
        solution = root(residual, z / zscale, tol=1e-10)
        assert solution.success
        p, t, _ = solution.x * zscale
        return inventory[0] / node.fluid_properties.state_pt("water", p, t).rho
    eps = 1e-3
    numerical = (volume_at(x + eps * direction) - volume_at(x - eps * direction)) / (2 * eps)
    assert trial.evaluation_data["volume_gradient"] @ direction == pytest.approx(numerical, rel=2e-7)


def test_wet_tank_mass_energy_and_interface_work_close_together():
    node, values = wet_tank()
    trial = node.evaluate(values)
    adjacent = [(-1, branch(trial.fluids["water"], 1.2)),
                (1, branch(trial.fluids["Nitrogen"], 0.03))]
    heat = {"liquid": 20., "gas": -10.}
    names = node.differential_variable_names
    def balance(vector):
        return node.residual(trial, dict(zip(names, vector)), adjacent, heat_rate=heat)[:4]
    zero = balance(np.zeros(4))
    # Evaluate the homogeneous rate operator directly; subtracting large
    # enthalpy fluxes to recover unit-energy columns loses precision.
    matrix = np.column_stack([node.residual(trial, dict(zip(names, v)))[:4]
                              for v in np.eye(4)])
    rates = np.linalg.solve(matrix, -zero)
    np.testing.assert_allclose(balance(rates), 0, atol=2e-8)
    assert rates[0] == pytest.approx(-1.2)
    assert rates[1] == pytest.approx(0.03)
    enthalpy = -1.2 * trial.fluids["water"]["h"] + 0.03 * trial.fluids["Nitrogen"]["h"]
    assert rates[2] + rates[3] == pytest.approx(enthalpy + sum(heat.values()), abs=2e-7)
    # Algebraic ydot entries do not affect the rate equations.
    given = {**dict(zip(names, rates)), "P": 1e20, "T_liq": -1e20, "T_ull": 1e20}
    np.testing.assert_allclose(node.residual(trial, given, adjacent, heat_rate=heat), 0, atol=2e-8)


def test_dryout_events_are_pure_and_modes_require_explicit_state_mapping():
    node, values = wet_tank()
    near = node.evaluate({**values, "m_liq": node.dry_mass})
    assert node.event_values(near)["dryout"] == 0
    assert node.event_values(near)["dryout"] == 0 and node.mode == "two_phase"
    assert node.set_mode("gas") and not node.set_mode("gas")
    with pytest.raises(ValueError, match="mapped by the network"):
        node.initial_values()
    # This test supplies an already gas-only state; set_mode does not discard
    # or convert the tiny liquid remainder. The future event mapper owns that.
    gas = node.fluid_properties.state_pt("Nitrogen", values["P"], values["T_ull"])
    dry = node.evaluate({"P": gas.P, "T": gas.T, "m": gas.rho, "U": gas.rho * gas.u})
    assert list(node.outlet(dry, port="liquid")) == ["Nitrogen"]
    assert node.output_state(dry)["mass"] == pytest.approx(gas.rho)
    with pytest.raises(ValueError, match="requires variables"):
        node.evaluate(values)


def test_saturated_volume_equilibrium_and_events_preserve_inventories():
    class Saturated(AnalyticProperties):
        def supports_saturation(self, fluid):
            return True
        def saturation_bounds(self, fluid):
            return 1e5, 1e6
        def saturation_at_p(self, fluid, pressure):
            return SaturationProperties(pressure, 300., self.state_pt("water", pressure, 300.),
                                        self.state_pt("Nitrogen", pressure, 300.))
    props = Saturated()
    sat = props.saturation_at_p("Nitrogen", 2e5)
    q = 0.3
    m = 1 / ((1 - q) / sat.liquid.rho + q / sat.vapor.rho)
    energy = m * ((1 - q) * sat.liquid.u + q * sat.vapor.u)
    node = VolumeComponent("volume", {"fluid": "Nitrogen", "geometry": Geometry(),
        "state0": {"P": 2e5, "quality": q, "m": m, "U": energy}},
        fluid_properties=props, phase="saturated")
    values = node.initial_values()
    trial = node.evaluate(values)
    np.testing.assert_allclose(node.residual(trial, {"m": 0., "U": 0.}), 0, atol=1e-10)
    assert node.outlet(trial)["Nitrogen"].phase == "gas"
    assert node.event_values(trial)["evaporate"] == 0.7
    assert node.event_values(trial)["liquid_limit"] == 0.3
    assert set(node.output_state(trial)["phase_states"]) == {"liquid", "gas"}
    node.set_mode("gas")
    trial = node.evaluate({"m": m, "U": energy, "P": 2e5, "T": 300.})
    assert node.event_values(trial)["condense"] == 0
    assert node.mode == "gas"  # Evaluating roots never arms or changes a mode.


def test_combustion_and_shutdown_match_existing_physics_without_mode_mutation():
    class Combustion:
        def evaluate(self, **kwargs):
            return CombustionProperties(1500., 1.5, 350., 1.2, 3000.)
    definition = dict(P0=2e6, oxidizer_fluid="ox", fuel_fluid="fuel", combustion_fluid="products",
                      ambient_node="ambient", expansion_ratio=5., cstar_efficiency=1., cf_efficiency=1.)
    props = Combustion()
    node = CombustorComponent("chamber", definition, combustion_properties=props)
    fluids = {name: FluidState(name, "liquid", {"h": 1e5, "rho": 1000.}) for name in ("ox", "fuel")}
    adjacent = [(1, branch(fluids["ox"], 2.)), (1, branch(fluids["fuel"], 1.))]
    ambient = {"ambient": {"P": 1e5}}
    trial = node.evaluate(node.initial_values(), adjacent, ambient)
    for key, value in dict(P=2e6, cstar=1500., Cf=1.5, MR=2., mdot_oxidizer=2., mdot_fuel=1.).items():
        assert trial[key] == value
    assert trial.fluids["products"]["T"] == 3000.
    adjacent.append((-1, branch(trial.fluids["products"], 3.)))
    np.testing.assert_allclose(node.residual(trial, {}, adjacent), 0)
    assert node.event_values(trial) == {"oxidizer_unavailable": 2., "fuel_unavailable": 1.}
    missing = node.evaluate({"P": 2e6}, [], ambient, reference_state=trial)
    assert missing["cstar"] == 0 and missing.fluids["products"]["T"] == 3000.
    assert node.mode == "combusting"
    node.set_mode("shutdown", ("liquid", "gas"), reason="fuel_unavailable")
    gas = FluidState("Nitrogen", "gas", {"h": 100., "T": 300., "R": 300., "gamma": 1.4})
    sources = [(1, branch(gas, 0.2)), (1, branch(fluids["ox"], 0.3))]
    shutdown = node.evaluate({"P": 2e5}, sources, ambient)
    assert set(node.outlet(shutdown)) == {"Nitrogen", "ox"}
    assert set(node.outlet(shutdown, port="products")) == {"Nitrogen", "ox"}
    assert node.outlet(shutdown, port="oxidizer")["ox"].phase == "liquid"
    sources += [(-1, branch(gas, 0.2)), (-1, branch(fluids["ox"], 0.3))]
    np.testing.assert_allclose(node.residual(shutdown, {}, sources), 0)
    assert node.event_values(shutdown) == {}
    assert node.output_state(shutdown)["shutdown_reason"] == "fuel_unavailable"
    node.set_mode("shutdown")
    empty = node.evaluate({}, [], ambient)
    assert empty["P"] == 1e5 and node.residual(empty, {}).size == 0


def test_table_derivatives_follow_interpolant_interior_knots_and_edges():
    p, t = np.array([1., 3., 6.]), np.array([10., 20., 40.])
    values = p[:, None]**2 + 2 * p[:, None] * t[None, :] + 3 * t[None, :]
    table = LookupTable("test", ("pressure", "temperature"), {"pressure": p, "temperature": t},
                        {"density": values}, {}, {}, {}, {}, {"success": np.ones((3, 3), bool)})
    for pressure, slope in [(1., 4.), (2., 4.), (3., 9.), (6., 9.)]:
        actual = table.derivative(["density"], wrt="pressure", pressure=pressure, temperature=15.)
        assert actual["density"] == pytest.approx(slope + 30.)
    assert table.derivative(["density"], wrt="temperature", pressure=2., temperature=20.)["density"] == 7.
    with pytest.raises(ValueError, match="outside"):
        table.derivative(["density"], wrt="pressure", pressure=0., temperature=15.)
    table.status["success"][1, 0] = False
    table._interpolate.cache_clear()
    with pytest.raises(ValueError, match="unsuccessful"):
        table.derivative(["density"], wrt="pressure", pressure=1., temperature=10.)


@pytest.mark.parametrize("fluid,p,t", [("Nitrogen", 2e6, 300.), ("Oxygen", 2e6, 95.)])
def test_coolprop_derivatives_match_direct_state_perturbations(fluid, p, t):
    source = CoolPropPropertySource()
    slopes = source.derivatives_pt(fluid, p, t)
    for suffix, delta in [("P", 10.), ("T", 1e-3)]:
        low = source.state_pt(fluid, p - delta if suffix == "P" else p, t - delta if suffix == "T" else t)
        high = source.state_pt(fluid, p + delta if suffix == "P" else p, t + delta if suffix == "T" else t)
        for name in ("rho", "u"):
            expected = (getattr(high, name) - getattr(low, name)) / (2 * delta)
            assert slopes[f"d{name}_d{suffix}"] == pytest.approx(expected, rel=2e-5, abs=1e-9)



@pytest.mark.parametrize("fluid,table_name,t", [("Nitrogen", "nitrogen_pt", 300.), ("Oxygen", "oxygen_pt", 95.)])
def test_real_table_provider_derivatives_match_its_property_values(fluid, table_name, t):
    source = TablePureFluidPropertySource(Path(__file__).resolve().parents[1] / "FluidTables/sizer_lookups.h5",
                                         {fluid: table_name})
    table = source._table(fluid)
    # Mid-cell probes avoid crossing a legitimate slope discontinuity.
    points = {}
    for axis_name, desired in (("pressure", 2e6), ("temperature", t)):
        axis = table.axes[axis_name]
        i = int(np.searchsorted(axis, desired))
        points[axis_name] = float((axis[i - 1] + axis[i]) / 2)
    p, t = points["pressure"], points["temperature"]
    slopes = source.derivatives_pt(fluid, p, t)
    for suffix, delta in (("P", 10.), ("T", 1e-3)):
        low = source.state_pt(fluid, p - delta if suffix == "P" else p, t - delta if suffix == "T" else t)
        high = source.state_pt(fluid, p + delta if suffix == "P" else p, t + delta if suffix == "T" else t)
        for name in ("rho", "u"):
            assert slopes[f"d{name}_d{suffix}"] == pytest.approx(
                (getattr(high, name) - getattr(low, name)) / (2 * delta), rel=2e-6, abs=1e-10)


@pytest.mark.parametrize("tables", [False, True])
def test_real_oxygen_nitrogen_tank_closes_with_shared_vehicle_geometry(tables):
    from Vehicle.tank_geometry import PropTankGeometry
    if tables:
        source = TablePureFluidPropertySource(Path(__file__).resolve().parents[1] / "FluidTables/sizer_lookups.h5",
                                             {"Nitrogen": "nitrogen_pt", "Oxygen": "oxygen_pt"})
    else:
        source = CoolPropPropertySource()
    liquid = source.state_pt("Oxygen", 2e6, 95.)
    gas = source.state_pt("Nitrogen", 2e6, 300.)
    ml, mg = 0.025 * liquid.rho, 0.015 * gas.rho
    node = PropellantTankComponent("ox_tank", {
        "liquid_fluid": "Oxygen", "gas_fluid": "Nitrogen",
        "geometry": PropTankGeometry(0.04, 0.3, 0.5, 1.75, 0., 64),
        "state0": dict(P=2e6, T=95., gas_T=300., m_liq=ml, m_ull=mg,
                       U_liq=ml * liquid.u, U_ull=mg * gas.u)}, fluid_properties=source)
    trial = node.evaluate(node.initial_values(), axial_specific_force=20.)
    rates = {name: 0. for name in node.differential_variable_names}
    np.testing.assert_allclose(node.residual(trial, rates), 0., atol=1e-9)
    assert np.all(np.isfinite(trial.evaluation_data["volume_gradient"]))
    events = node.event_values(trial)
    assert all(value > 0 for value in events.values())
    assert sum(node.output_state(trial)["axial_mass"]) == pytest.approx(ml + mg)
    with pytest.raises(TrialDomainError, match="positive liquid"):
        node.evaluate({**node.initial_values(), "m_liq": 0.})


def test_nodes_share_trial_error_type_and_keep_assembly_errors_distinct():
    from diagnostics.errors import TrialDomainError as SharedError
    assert TrialDomainError is SharedError
    node, values = wet_tank()
    for changed in ({"P": -1.}, {"m_liq": 0.}, {"U_liq": np.nan}):
        with pytest.raises(TrialDomainError):
            node.evaluate({**values, **changed})
    trial = node.evaluate(values)
    with pytest.raises(TrialDomainError, match="non-finite derivative"):
        node.residual(trial, {name: np.nan for name in node.differential_variable_names})
    for action in (
        lambda: node.evaluate({}),
        lambda: node.residual(trial, {}),
        lambda: node.set_mode("unknown"),
        lambda: BoundaryComponent("missing", {}).evaluate({}),
    ):
        with pytest.raises(ValueError) as error:
            action()
        assert type(error.value) is ValueError


def test_algebraic_response_matches_high_precision_closures_without_matrix_solve(monkeypatch):
    from decimal import Decimal, localcontext

    def forbidden(*args, **kwargs):
        raise AssertionError('The tank response must not use a numerical fallback')
    monkeypatch.setattr(np.linalg, 'solve', forbidden)
    source = TablePureFluidPropertySource(Path(__file__).resolve().parents[1] / 'FluidTables/sizer_lookups.h5',
                                        {'Oxygen': 'oxygen_pt', 'Nitrogen': 'nitrogen_pt'})
    rng = np.random.default_rng(42)
    for _ in range(60):
        p, tl, tg = rng.uniform(5e5, 4e6), rng.uniform(85, 100), rng.uniform(240, 330)
        liquid, gas = source.state_pt('Oxygen', p, tl), source.state_pt('Nitrogen', p, tg)
        dl, dg = source.derivatives_pt('Oxygen', p, tl), source.derivatives_pt('Nitrogen', p, tg)
        ml, mg = 10**rng.uniform(-12, 3), 10**rng.uniform(-12, 1)
        volume, pressure = PropellantTankComponent._thermodynamic_response(ml, mg, liquid, gas, dl, dg)
        # Independent 60-digit elimination of the differentiated EOS equations.
        # This avoids treating either floating-point implementation as exact.
        with localcontext() as ctx:
            ctx.prec = 60
            D = lambda v: Decimal(float(v))
            vl = [-D(ml)*D(dl[k])/D(liquid.rho)**2 for k in ('drho_dP', 'drho_dT')]
            vg = [-D(mg)*D(dg[k])/D(gas.rho)**2 for k in ('drho_dP', 'drho_dT')]
            rows = [[D(ml)*D(dl['du_dP']), D(ml)*D(dl['du_dT']), D(0), -D(liquid.u), D(0), D(1), D(0)],
                    [D(mg)*D(dg['du_dP']), D(0), D(mg)*D(dg['du_dT']), D(0), -D(gas.u), D(0), D(1)],
                    [vl[0]+vg[0], vl[1], vg[1], -1/D(liquid.rho), -1/D(gas.rho), D(0), D(0)]]
            for i in range(3):
                pivot = max(range(i, 3), key=lambda j: abs(rows[j][i]))
                rows[i], rows[pivot] = rows[pivot], rows[i]
                divisor = rows[i][i]
                rows[i] = [v/divisor for v in rows[i]]
                for j in range(3):
                    if j != i:
                        factor = rows[j][i]
                        rows[j] = [a-factor*b for a, b in zip(rows[j], rows[i])]
            expected_p = [float(v) for v in rows[0][3:]]
            expected_v = [float(vl[0]*rows[0][3+i] + vl[1]*rows[1][3+i]
                                + (1/D(liquid.rho) if i == 0 else 0)) for i in range(4)]
        np.testing.assert_allclose(pressure, expected_p, rtol=2e-11, atol=1e-15)
        np.testing.assert_allclose(volume, expected_v, rtol=2e-11, atol=1e-20)


@pytest.mark.parametrize('failure', ['zero_mass', 'nonfinite', 'zero_heat_derivative', 'singular', 'near_singular'])
def test_algebraic_response_reports_invalid_trials(failure):
    node, values = wet_tank()
    props, p = node.fluid_properties, values['P']
    liquid, gas = props.state_pt('water', p, values['T_liq']), props.state_pt('Nitrogen', p, values['T_ull'])
    dl, dg = props.derivatives_pt('water', p, values['T_liq']), props.derivatives_pt('Nitrogen', p, values['T_ull'])
    ml, mg = values['m_liq'], values['m_ull']
    if failure == 'zero_mass': ml = 0.
    if failure == 'nonfinite': dl['du_dP'] = float('nan')
    if failure == 'zero_heat_derivative': dg['du_dT'] = 0.
    if failure == 'singular':
        dl['drho_dP'] = dl['drho_dT'] = dg['drho_dP'] = dg['drho_dT'] = 0.
    if failure == 'near_singular':
        dl['drho_dT'] = dg['drho_dT'] = 0.
        dl['drho_dP'] = -mg * dg['drho_dP'] / gas.rho**2 * liquid.rho**2 / ml * (1-1e-12)
    with pytest.raises(TrialDomainError, match='thermodynamic response'):
        node._thermodynamic_response(ml, mg, liquid, gas, dl, dg)


def test_junction_mixing_energy_and_zero_flow():
    props = AnalyticProperties()
    node = JunctionComponent('mix', dict(fluid='Nitrogen', phase='gas',
        state0=dict(P=2e5,T=300.)), fluid_properties=props)
    cold = FluidState('Nitrogen','gas',props.state_pt('Nitrogen',3e5,300.).as_dict())
    hot = FluidState('Nitrogen','gas',props.state_pt('Nitrogen',4e5,600.).as_dict())
    incoming = [(1,branch(cold,2.)), (1,branch(hot,1.))]
    mixed = node.evaluate(dict(P=2e5,T=400.), incoming)
    outgoing = (-1,branch(mixed.fluids['Nitrogen'],3.))
    for donors in (incoming, incoming[::-1]):
        np.testing.assert_allclose(node.residual(mixed,{},[*donors,outgoing]),[0,0],atol=1e-8)
    wrong = node.evaluate(dict(P=2e5,T=500.), incoming)
    assert abs(node.residual(wrong,{},[*incoming,(-1,branch(wrong.fluids['Nitrogen'],3.))])[1]) > 1e5
    # Isolated P/T are prescribed, with no fictitious inventory or flow.
    isolated = node.evaluate(node.initial_values())
    np.testing.assert_allclose(node.residual(isolated,{}),[0,0])
    assert node.differential_variable_names == ()
    with pytest.raises(ValueError,match='cannot store heat'):
        node.residual(isolated,{},heat_rate={'gas':1.})
    different = FluidState('water','liquid',props.state_pt('water',2e5,300.).as_dict())
    with pytest.raises(ValueError,match='different fluids or phases'):
        node.select_fluid([*incoming,(1,branch(different,1.))])


def test_junction_requires_explicit_initial_state():
    with pytest.raises(KeyError):
        JunctionComponent('mix',dict(fluid='Nitrogen',phase='gas',P0=2e5),
                          fluid_properties=AnalyticProperties())
