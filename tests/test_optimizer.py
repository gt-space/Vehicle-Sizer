from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import optimizer as opt
from Configs.loader import load_config
from constraints import ConstraintRecord, EvaluationFailure, finalize
from simulation_types import SimResult


@pytest.fixture
def settings():
    settings = load_config('Configs/optimizer.yaml')
    # Scoring examples use a fixed reference independent of user search tuning.
    settings['reference_mass'] = 500.0
    settings['conditional_geometry'] = False
    settings.pop('soft_penalties', None)
    return settings


def accepted(mass=500):
    return finalize(SimResult(initial_mass=mass, apogee=150000, burn_complete=True,
                              burn_duration=45), {'goal_apogee': 150000, 'max_burn_duration': 45})


def test_score_feasibility_first_and_scaled_deficits(settings):
    assert opt.candidate_score(accepted(500), settings)[0] == .5
    assert opt.candidate_score(accepted(250), settings)[0] < .5
    result = finalize(SimResult(apogee=145000, burn_complete=True, burn_duration=47,
                               constraints={'tank.press.Tmin': -2}),
                      {'goal_apogee': 150000, 'max_burn_duration': 45})
    score, parts = opt.candidate_score(result, settings)
    assert score == pytest.approx(1 + 1.1/2.1)
    assert parts == {'tank.press.Tmin': .2, 'goal_apogee': .5, 'max_burn_duration': .4}


def test_missing_and_preflight_do_not_fabricate_flight_penalties(settings):
    limits = {'goal_apogee': 150000, 'max_burn_duration': 45}
    preflight = finalize(SimResult(termination='infeasible_initial_design',
                                  geometry_constraints={'engine.length': -.005}), limits)
    assert opt.candidate_score(preflight, settings) == (2 + .5/1.5, {'engine.length': .5})
    incomplete = finalize(SimResult(burn_complete=True, burn_duration=40, max_altitude=160000), limits)
    assert opt.candidate_score(incomplete, settings) == (3, {})
    incomplete.burn_complete = False
    incomplete.constraints['max_burn_duration'] = None
    assert opt.candidate_score(incomplete, settings)[0] == 3


def test_optional_and_redundant_checks_do_not_add_penalties(settings):
    result = opt.rejection({
        'optional': ConstraintRecord(None, '', 'sizing', 'm', required=False),
        'tank.press.Pmin': ConstraintRecord(-1e6, '', 'sizing', 'Pa'),
        'branch.feed.choked': ConstraintRecord(-1e6, '', 'sizing', 'Pa'),
    })
    assert opt.candidate_score(result, settings, redundant={'branch.feed.choked'})[0] == 2.5
    settings['unit_scales']['Pa'] = 0
    with pytest.raises(ValueError, match='scale'):
        opt.candidate_score(result, settings)


def test_decode_changes_all_19_inputs_without_mutating_fixed_choices(settings, tmp_path):
    cfg = load_config(settings['base_config'])
    cfg['tanks']['press_tank']['volume'] = cfg['tanks']['press_tank'].pop('volume_liters') * .001 if 'volume_liters' in cfg['tanks']['press_tank'] else cfg['tanks']['press_tank']['volume']
    before = deepcopy(cfg)
    evaluator = opt.Evaluator(cfg, settings, None, None, None, tmp_path)
    x = np.mean(evaluator.bounds, axis=1)
    candidate = evaluator.decode(x)
    assert len(x) == 19
    for value, path in zip(x, evaluator.paths.values()):
        assert opt.get_path(candidate, path) == value
    assert cfg == before
    assert candidate['aero'] == cfg['aero']
    assert candidate['engine'] == {**cfg['engine'], 'exit_pressure': candidate['engine']['exit_pressure']}
    assert candidate['engine']['exit_pressure'] != cfg['engine']['exit_pressure']
    assert candidate['vehicle']['sections'] == cfg['vehicle']['sections']
    assert candidate['tanks']['press_tank']['outer_diameter'] == cfg['tanks']['press_tank']['outer_diameter']


def test_real_deck_bounds_preparation_and_derived_domain(settings):
    model = opt.DragModel()
    cfg, prepared = opt.prepare(settings, model)
    assert opt.aero_bounds(model)['omld'] == pytest.approx((.2032, .4064))
    assert cfg['simulation']['t_end'] == 600
    assert 'P' not in cfg['prop_system']['initial_conditions']['ox_tank']
    assert 'mass' not in cfg['tanks']['press_tank']
    bad = deepcopy(settings)
    bad['bounds']['span'][1] = 1
    with pytest.raises(ValueError, match='aero deck'):
        opt.prepare(bad, model)
    v = dict(omld=12, length=501, fineness=5, exit=11, boattail_aft=10,
             boattail_length=23, span=7, root=18, tip=6, sweep_fraction=.75, thickness=.375)
    records = opt.domain_records(v, model)
    assert records['aero.length'].margin == pytest.approx(-.0254)
    assert records['aero.exit'].margin == pytest.approx(-.0254)
    assert records['aero.omld'].margin > 0


def test_primitive_rejection_skips_expensive_models(settings, tmp_path):
    cfg = load_config(settings['base_config'])
    cfg['tanks']['press_tank']['volume'] = .02
    evaluator = opt.Evaluator(cfg, settings, None, None, None, tmp_path)
    cfg['fin_can']['boattail_length'] = .51
    with patch.object(opt, 'Vehicle', side_effect=AssertionError('must skip build')):
        result, _ = evaluator.evaluate(cfg)
    assert result.termination == 'infeasible_initial_design'
    assert result.constraints['engine.length'] < 0


def test_logging_best_budget_and_unresolved_failure(settings, tmp_path):
    cfg = load_config(settings['base_config'])
    cfg['tanks']['press_tank']['volume'] = .02
    settings['max_evaluations'] = 2
    evaluator = opt.Evaluator(cfg, settings, None, None, None, tmp_path)
    x = np.mean(evaluator.bounds, axis=1)
    with patch.object(evaluator, 'evaluate', return_value=(accepted(), set())):
        assert evaluator(x) == .5
    assert (tmp_path / 'best.yaml').exists()
    failure = EvaluationFailure('runtime', cfg)
    with patch.object(evaluator, 'evaluate', side_effect=failure):
        assert evaluator(x) == 3.
    assert (tmp_path / 'failed_0002.yaml').exists()
    assert 'EvaluationFailure' in (tmp_path / 'failure_0002.txt').read_text()
    with pytest.raises(opt.EvaluationBudget):
        evaluator(x)
    rows = (tmp_path / 'evaluations.jsonl').read_text().splitlines()
    assert len(rows) == 2 and json.loads(rows[0])['accepted']
    assert json.loads(rows[1])['score_class'] == 'unresolved'


def test_incumbent_uses_soft_objective_not_mass(settings, tmp_path):
    cfg = load_config(settings['base_config'])
    cfg['tanks']['press_tank']['volume'] = .02
    settings['soft_penalties'] = {'max_q': dict(weight=.1, target_pa=140000., scale_pa=10000.)}
    evaluator = opt.Evaluator(cfg, settings, None, None, None, tmp_path)
    x = np.mean(evaluator.bounds, axis=1)
    light = accepted(250.)
    light.max_q = 160000.
    heavy = accepted(275.)
    heavy.max_q = 140000.
    with patch.object(evaluator, 'evaluate', side_effect=[(light, set()), (heavy, set())]):
        first = evaluator(x)
        assert evaluator(x) < first
    assert evaluator.best['mass'] == 275.


def test_pressure_fed_policy_validates_owned_limits():
    settings = load_config('Configs/optimizer_pressure_fed.yaml')
    assert 'constraints' not in load_config(settings['base_config'])
    model = opt.DragModel()
    cfg, prepared = opt.prepare(settings, model)
    assert cfg['constraints']['min_stability_calibers'] == 2.
    assert cfg['constraints']['max_length_to_diameter'] == 25.
    assert cfg['constraints']['goal_apogee'] == 150000.
    assert cfg['constraints']['max_burn_duration'] == 75.
    assert 'constraints' not in load_config(settings['base_config'])
    assert cfg['tanks']['fuel_tank']['min_temperature'] == 80.
    assert prepared['soft_penalties']['max_q']['target_pa'] == 140000.
    bad = deepcopy(settings)
    bad['hard_constraints'].append('tank.missing.Pmin')
    with pytest.raises(ValueError, match='requires a limit'):
        opt.prepare(bad, model)


@pytest.mark.parametrize('kind', ['table', 'solver', 'residual', 'unexpected', 'configuration', 'io'])
def test_typed_failures_are_reported_or_stop(settings, tmp_path, kind):
    from errors import LookupBoundsError, SolverConvergenceError, ResidualAcceptanceError
    causes = dict(table=LookupBoundsError('outside'),
                  solver=SolverConvergenceError('integration', -4, 1.),
                  residual=ResidualAcceptanceError('closure'), unexpected=RuntimeError('unknown'),
                  configuration=ValueError('invalid setting'), io=PermissionError('locked'))
    cfg = load_config(settings['base_config'])
    evaluator = opt.Evaluator(cfg, settings, None, None, None, tmp_path)
    x = np.mean(evaluator.bounds, axis=1)
    failure = EvaluationFailure('runtime', cfg)
    failure.__cause__ = causes[kind]
    with patch.object(evaluator, 'evaluate', side_effect=failure):
        if kind in ('configuration', 'io'):
            with pytest.raises(EvaluationFailure):
                evaluator(x)
            assert not (tmp_path/'evaluations.jsonl').exists()
        else:
            assert evaluator(x) == 3.
            entry = json.loads((tmp_path/'evaluations.jsonl').read_text())
            expected = dict(table='table_domain_exceeded', solver='solver_nonconvergence',
                            residual='residual_acceptance_failure', unexpected='unexpected_error')
            assert entry['termination'] == entry['failure_details']['kind'] == expected[kind]


def test_real_scipy_budget_and_no_false_winner(settings, tmp_path):
    settings['max_evaluations'] = 3
    result = opt.rejection({'engine.length': ConstraintRecord(-.01, '', 'sizing', 'm')})
    with patch.object(opt, 'property_sources', return_value=(None, None)), \
         patch.object(opt.Evaluator, 'evaluate', return_value=(result, set())):
        summary = opt.optimize(settings, tmp_path / 'run')
    assert summary == dict(evaluations=3, stop='evaluation budget exhausted', accepted=False,
                           search_accepted=False, verified=None, best_mass=None)
    assert not (tmp_path / 'run' / 'best.yaml').exists()
    with pytest.raises(FileExistsError):
        opt.optimize(settings, tmp_path / 'run')


@pytest.mark.parametrize('passes', [True, False])
def test_verification_can_reject_search_winner(settings, tmp_path, passes):
    cfg = load_config(settings['base_config'])
    (tmp_path / 'best.yaml').write_text(opt.yaml.safe_dump(cfg))
    evaluator = opt.Evaluator(cfg, settings, None, None, None, tmp_path)
    result = accepted()
    if not passes:
        result.apogee = None
    with patch.object(opt, 'simulate', return_value=result) as simulate:
        assert opt.verify_best(evaluator) is passes
    call = simulate.call_args
    assert call.args[0]['simulation']['dt'] == cfg['simulation']['dt'] * .5
    assert call.kwargs['compute_loads'] is True
    assert (tmp_path / 'verified.yaml').exists() is passes
    assert json.loads((tmp_path / 'verification.json').read_text())['accepted'] is passes


@pytest.mark.parametrize('mode,suffix', [('bang_bang', 'BANGBANG'), ('regulator', 'REGULATOR')])
def test_no_implicit_choking_penalties(settings, mode, suffix):
    cfg = load_config(settings['base_config'])
    cfg = load_config(f'Configs/flight_pump_fed_{mode}.yaml')
    assert opt.Evaluator.redundant(None, cfg) == set()
    cfg['prop_system']['template'] = {
        'circuits': {'gas': {}},
        'nodes': {'gas': {'component': 'pressurant_tank', 'tank_id': 'supply'}, 'boundary': {'component': 'boundary'}},
        'branches': {'feed': {'component': 'loss', 'circuit': 'gas', 'from': 'gas', 'to': 'boundary'},
                     'standalone': {'component': 'loss', 'circuit': 'gas', 'from': 'boundary', 'to': 'gas'}}}
    assert opt.Evaluator.redundant(None, cfg) == set()


@pytest.mark.parametrize('length', [360, 501])
def test_sized_aero_domain_gates_headless_flight(settings, tmp_path, length):
    cfg = load_config(settings['base_config'])
    cfg['tanks']['press_tank']['volume'] = .02
    model = SimpleNamespace(ranges={'length': (276, 500)})
    vehicle = SimpleNamespace(tanks={}, build=lambda engine: None,
                              aero_candidate=lambda: {'length': length})
    propulsion = SimpleNamespace(exit_area=.02, choked_branches={}, node_definitions={})
    evaluator = opt.Evaluator(cfg, settings, model, 'pure', 'combustion', tmp_path)
    with patch.object(opt, 'Vehicle', return_value=vehicle), \
         patch.object(opt, 'PropSystem', return_value=propulsion), \
         patch.object(opt, 'simulate', return_value=accepted()) as simulate:
        result, _ = evaluator.evaluate(cfg)
    if length == 501:
        simulate.assert_not_called()
        assert result.constraints['aero.length'] < 0
    else:
        assert result.accepted
        simulate.assert_called_once_with(cfg, pure_properties='pure',
                                          combustion_properties='combustion', aero_model=model)


def test_class_order_dominates_violation_size(settings):
    complete = finalize(SimResult(apogee=1000, burn_complete=True, burn_duration=500,
                                 constraints={'tank.press.Tmin': -1000}),
                        {'goal_apogee': 150000, 'max_burn_duration': 45})
    tiny_geometry = opt.rejection({'engine.length': ConstraintRecord(-1e-8, '', 'sizing', 'm')})
    scores = [opt.candidate_score(r, settings)[0] for r in (accepted(1e9), complete, tiny_geometry, SimResult())]
    assert scores[0] < 1 <= scores[1] < 2 <= scores[2] < 3 == scores[3]
    complete.constraint_records['required_missing'] = ConstraintRecord(None, '', 'runtime', 'Pa')
    assert opt.candidate_score(complete, settings)[0] == 3


def test_conditional_geometry_matches_actual_sizing_and_user_bounds(settings, tmp_path):
    settings['conditional_geometry'] = True
    model = opt.DragModel()
    cfg, settings = opt.prepare(settings, model)
    pure, combustion = opt.property_sources(cfg)
    evaluator = opt.Evaluator(cfg, settings, model, pure, combustion, tmp_path)
    rng = np.random.default_rng(42)
    bounds = np.array(evaluator.bounds)
    count = len(evaluator.paths)
    for unit in [np.zeros(count), np.ones(count), *rng.random((10, count))]:
        candidate = evaluator.decode(bounds[:, 0] + unit*(bounds[:, 1]-bounds[:, 0]))
        for name, path in evaluator.paths.items():
            assert settings['bounds'][name][0] <= opt.get_path(candidate, path) <= settings['bounds'][name][1]
        assert candidate['fin_can']['root_chord'] <= candidate['fin_can']['boattail_length']
        assert candidate['fin_can']['boattail_aft_diameter'] <= candidate['vehicle']['OMLD']
        # Avoid pump infeasibility unrelated to the geometry under test.
        candidate['prop_system']['pumps']['oxidizer_pump']['pressure_rise_pa'] = 5e5
        candidate['prop_system']['pumps']['fuel_pump']['pressure_rise_pa'] = 5e5
        vehicle = opt.Vehicle(candidate, pure)
        prop = opt.PropSystem(candidate, vehicle.tanks, pure, combustion)
        vehicle.build(prop)
        assert vehicle.geometry_constraints['engine.nozzle'] >= 0
        assert vehicle.geometry_constraints['engine.length'] >= 0
        assert vehicle.geometry_constraints['press_tank.airframe'] >= 0


def test_conditional_geometry_rejects_empty_user_interval(settings, tmp_path):
    settings['conditional_geometry'] = True
    settings['bounds']['boattail_length'] = [.508, .55]
    model = opt.DragModel()
    cfg, settings = opt.prepare(settings, model)
    pure, combustion = opt.property_sources(cfg)
    evaluator = opt.Evaluator(cfg, settings, model, pure, combustion, tmp_path)
    with pytest.raises(opt.GeometryError):
        evaluator.decode(np.mean(evaluator.bounds, axis=1))


@pytest.mark.parametrize('flat', [True, False])
@pytest.mark.parametrize('generations', [0, 3])
def test_generation_budget_does_not_converge_on_rejections(flat, generations):
    def score(x):
        return 3. if flat else 2.99 + 1e-6 * sum(x)
    with opt.GenerationBudgetSolver(score, [(0.,1.)]*2, rng=42,
            popsize=4, maxiter=generations, polish=False) as solver:
        result = solver.solve()
    assert result.nit == generations
    assert result.nfev == 8 * (generations + 1)
    assert not result.success  # Exhausting a budget is not convergence.


def test_evaluation_budget_interrupts_a_generation():
    calls = []
    def score(x):
        if len(calls) == 11:
            raise opt.EvaluationBudget
        calls.append(x.copy())
        return 3.
    with opt.GenerationBudgetSolver(score, [(0.,1.)]*2, rng=42,
            popsize=4, maxiter=100, polish=False) as solver:
        with pytest.raises(opt.EvaluationBudget):
            solver.solve()
    assert len(calls) == 11
