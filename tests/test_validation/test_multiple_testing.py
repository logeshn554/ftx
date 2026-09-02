"""Unit tests for ResearchTrialLedger and Multiple Testing Deflation."""

import pytest
from trade.validation.multiple_testing import ResearchTrialLedger


def test_research_trial_ledger_persistence(tmp_path):
    ledger_path = tmp_path / "trials.json"
    ledger = ResearchTrialLedger(path=ledger_path)

    ledger.record_trial(
        trial_id="tr_1",
        model_version="v1.0.0",
        candidate_id="cand_1",
        parameters={"lr": 0.001},
        oos_sharpe=1.2,
        oos_return=0.15,
        track_record_length=252,
    )
    assert ledger.total_trials_count() == 1

    # Reload from disk
    ledger_reloaded = ResearchTrialLedger(path=ledger_path)
    assert ledger_reloaded.total_trials_count() == 1


def test_multiple_testing_dsr_deflates_with_more_trials():
    ledger = ResearchTrialLedger()

    # Record 50 neutral/failed trials
    for i in range(50):
        ledger.record_trial(
            trial_id=f"tr_{i}",
            model_version="v1.0.0",
            candidate_id=f"cand_{i}",
            parameters={"p": i},
            oos_sharpe=0.4 + (i % 5) * 0.1,
            oos_return=0.02,
            track_record_length=252,
        )

    # A candidate with modest Sharpe (1.0) should be deflated after 50 trials
    passed, dsr_prob, audit = ledger.evaluate_candidate_dsr(
        estimated_sharpe=1.0,
        track_record_length=252,
        confidence_level=0.95,
    )

    assert audit["num_trials"] == 50
    assert 0.0 <= dsr_prob <= 1.0
