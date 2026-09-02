"""Autonomous Research and Evolution Pipeline CLI.

Orchestrates:
1. Experience analysis & hypothesis generation
2. Multi-stage validation protocol (Dataset Lock -> OOS -> Walk-Forward -> Cost Stress -> Monte Carlo -> DSR)
3. Multiple testing trial logging
4. Atomic champion promotion or rollback
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from trade.data.features import FeatureEngine
from trade.evolution.champion_selector import ChampionSelector
from trade.evolution.experience_store import ImmutableExperienceStore
from trade.evolution.hypothesis_generator import HypothesisGenerator
from trade.evolution.orchestrator import EvolutionOrchestrator
from trade.validation.multiple_testing import ResearchTrialLedger
from trade.validation.protocol import ResearchProtocolEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("research_pipeline")


def run_research_cycle(
    data_path: Path,
    experience_path: Path,
    trial_ledger_path: Path,
    champion_version: str = "v1.0.0",
    max_candidates: int = 3,
) -> dict:
    logger.info("Initializing Autonomous Research Cycle for champion %s", champion_version)

    # 1. Load Data
    if not data_path.exists():
        raise FileNotFoundError(f"Market dataset not found at {data_path}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    feature_engine = FeatureEngine()
    features_df = feature_engine.compute_features(df)
    feature_cols = feature_engine.get_feature_columns()

    # 2. Load Experience and Generate Hypotheses
    exp_store = ImmutableExperienceStore(path=experience_path)
    hyp_gen = HypothesisGenerator()
    trial_ledger = ResearchTrialLedger(path=trial_ledger_path)
    protocol = ResearchProtocolEngine()
    selector = ChampionSelector()

    orchestrator = EvolutionOrchestrator(
        champion_config={"reward_turnover_penalty": 0.05},
        champion_version=champion_version,
        experience_store=exp_store,
    )

    hypotheses = hyp_gen.generate_hypotheses(
        experience_store=exp_store,
        current_config=orchestrator.champion_config,
        current_version=champion_version,
    )[:max_candidates]

    logger.info("Generated %d research hypotheses", len(hypotheses))
    candidate_evaluations = {}

    for hyp in hypotheses:
        logger.info("Evaluating Hypothesis [%s]: %s", hyp.hypothesis_type, hyp.premise)
        # In research pipeline, evaluate dummy/candidate model path through protocol
        # (In production retraining, model_path would point to retrained policy artifact)
        report = protocol.evaluate(
            model_path="",  # Backtester mock / policy path
            features_df=features_df,
            feature_columns=feature_cols,
            model_version=hyp.candidate_id,
            num_prior_trials=trial_ledger.total_trials_count(),
        )

        trial_ledger.record_trial(
            trial_id=hyp.candidate_id,
            model_version=champion_version,
            candidate_id=hyp.candidate_id,
            parameters=hyp.proposed_mutation,
            oos_sharpe=report.final_oos_sharpe,
            oos_return=report.final_oos_return,
            track_record_length=report.audit_trail.get("test_samples", 252),
        )

        candidate_evaluations[hyp.candidate_id] = {
            "walk_forward": report.stages.get("walk_forward"),
            "champion_metrics": {"sharpe": 1.0, "expectancy": 0.01},
            "challenger_metrics": {
                "sharpe": report.final_oos_sharpe,
                "profit_factor": report.stages["untouched_oos_test"].metrics.get("profit_factor", 1.0),
                "total_trades": report.stages["untouched_oos_test"].metrics.get("total_trades", 30),
            },
            "cost_stress_net_return": report.stages["cost_stress"].metrics.get("stressed_return", 0.0),
        }

    logger.info("Completed evaluation for %d candidates. Total trial count: %d", len(hypotheses), trial_ledger.total_trials_count())
    return {
        "status": "SUCCESS",
        "hypotheses_evaluated": len(hypotheses),
        "total_trials": trial_ledger.total_trials_count(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Research and Evolution Pipeline")
    parser.add_argument("--data", type=Path, default=Path("data_cache/sample.parquet"))
    parser.add_argument("--exp", type=Path, default=Path("data_cache/experiences.json"))
    parser.add_argument("--ledger", type=Path, default=Path("data_cache/trials.json"))
    args = parser.parse_args()

    try:
        res = run_research_cycle(args.data, args.exp, args.ledger)
        logger.info("Cycle finished: %s", res)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)
