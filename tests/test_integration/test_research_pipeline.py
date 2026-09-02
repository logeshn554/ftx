"""Integration test for Autonomous Research Pipeline."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from scripts.run_research_pipeline import run_research_cycle


def test_autonomous_research_pipeline_cycle(sample_ohlcv, tmp_path):
    data_path = tmp_path / "data.parquet"
    sample_ohlcv.to_parquet(data_path)

    exp_path = tmp_path / "exp.json"
    ledger_path = tmp_path / "ledger.json"

    mock_model = MagicMock()
    mock_model.predict.return_value = (0, None)

    with patch("trade.validation.backtester.PPO.load", return_value=mock_model):
        res = run_research_cycle(
            data_path=data_path,
            experience_path=exp_path,
            trial_ledger_path=ledger_path,
            champion_version="v1.0.0",
            max_candidates=2,
        )

        assert res["status"] == "SUCCESS"
        assert res["hypotheses_evaluated"] >= 1
        assert res["total_trials"] >= 1
        assert ledger_path.exists()
