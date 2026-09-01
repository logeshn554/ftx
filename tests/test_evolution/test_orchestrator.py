from trade.core.types import ModelVersion
from trade.evolution.candidate_generator import CandidateGenerator
from trade.evolution.champion_selector import ChampionSelector
from trade.evolution.evaluator import CandidateEvaluator
from trade.evolution.orchestrator import EvolutionOrchestrator
from trade.evolution.rollback_snapshot import RollbackArchive, RollbackSnapshot
from trade.validation.walk_forward import WalkForwardResult


def test_one_loss_cannot_change_champion():
    orch = EvolutionOrchestrator({"training": {}}, champion_version="v1")
    original = orch.champion_version
    orch.on_trade_closed()
    assert orch.champion_version == original


def test_only_valid_promotion_changes_champion():
    orch = EvolutionOrchestrator({"training": {}}, champion_version="v1")
    gen = CandidateGenerator(seed=1)
    candidate = gen.generate(orch.champion_config, "v1", 1)[0]
    wf = WalkForwardResult(model_version=ModelVersion(0, 0, 1), n_windows=5, positive_window_ratio=0.8, oos_return_mean=0.05)
    evaluation = orch.evaluate_candidate(
        candidate, wf, {"sharpe_ratio": 1.0}, {"sharpe_ratio": 1.5, "profit_factor": 1.2, "total_trades": 50},
        cost_stress_net_return=1.0,
    )
    decision = orch.selector.decide(evaluation, 1.0, 1.5, 0.01, 0.02, 0.1, 1.2, 50)
    if decision.promote:
        orch.promote(evaluation, decision)
        assert orch.champion_version == candidate.candidate_id
    else:
        assert orch.champion_version == "v1"


def test_rollback_restores_champion():
    orch = EvolutionOrchestrator({"a": 1}, champion_version="v1")
    orch._rollback_config = {"a": 1}
    orch.champion_config = {"a": 2}
    assert orch.rollback()
    assert orch.champion_config == {"a": 1}


def test_rollback_archive_snapshot_restore():
    orch = EvolutionOrchestrator({"a": 10}, champion_version="v1.0.0")
    gen = CandidateGenerator(seed=1)
    candidate = gen.generate(orch.champion_config, "v1.0.0", 1)[0]
    wf = WalkForwardResult(model_version=ModelVersion(0, 0, 1), n_windows=5, positive_window_ratio=0.8, oos_return_mean=0.05)
    evaluation = orch.evaluate_candidate(
        candidate, wf, {"sharpe_ratio": 1.0}, {"sharpe_ratio": 1.5, "profit_factor": 1.2, "total_trades": 50},
        cost_stress_net_return=1.0,
    )
    decision = orch.selector.decide(evaluation, 1.0, 1.5, 0.01, 0.02, 0.1, 1.2, 50)
    
    promoted = orch.promote(evaluation, decision)
    assert promoted is True
    assert orch.champion_version == candidate.candidate_id
    
    # Rollback to pre-promotion snapshot
    assert orch.rollback() is True
    assert orch.champion_version == "v1.0.0"
    assert orch.champion_config == {"a": 10}


def test_run_full_evolution_pipeline():
    orch = EvolutionOrchestrator({"training": {"reward_turnover_penalty": 0.05}}, champion_version="v1.0.0")
    
    # When conditions are not met, evolution is safely skipped
    res_skipped = orch.run_full_evolution(count=2)
    assert res_skipped["status"] == "SKIPPED"
    
    # Trigger on degradation flag
    orch.should_trigger_evolution = lambda persistent_negative_expectancy=False: True
    res_completed = orch.run_full_evolution(count=2)
    assert res_completed["status"] == "COMPLETED"
    assert res_completed["candidates_evaluated"] == 2


def test_candidate_generator_immutable_champion():
    champion = {"training": {"reward_turnover_penalty": 0.05}}
    original = dict(champion)
    CandidateGenerator(seed=1).generate(champion, "v1", 2)
    assert champion == original
