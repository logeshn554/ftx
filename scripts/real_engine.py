"""Real Inference Engine for Self-Evolving Crypto Trader.

Loads the extracted artifacts:
1. Multi-timeframe Parquet dataset (BTCUSDT)
2. HMM Regime Classifier (GaussianHMM)
3. PPO Policy (RecurrentPPO with LSTM)
4. XGBoost Failure/Loss Analyzer
5. Experience Memory Vectors

Executes live sequential model inference and exposes state for streaming.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pickle
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from trade.data.contract import observation_columns
from trade.core.config import AppConfig  # FIX 18: Load config for thresholds

logger = logging.getLogger("real_engine")

BASE_DIR = Path(__file__).parent.parent.resolve()
EXTRACTED_DIR = BASE_DIR / "self_evolving_crypto"
MODELS_DIR = EXTRACTED_DIR / "models"
DATA_DIR = EXTRACTED_DIR / "data" / "processed"
MEMORY_DIR = EXTRACTED_DIR / "memory"

class RealInferenceEngine:
    def __init__(self, config_path: str | None = None):
        # FIX 18: Load configuration for decision thresholds
        if config_path is None:
            config_path = str(BASE_DIR / "config" / "default.yaml")
        
        self.config = AppConfig.from_yaml(config_path)
        self.min_action_confidence = self.config.intelligence.min_action_confidence
        self.max_loss_risk = self.config.intelligence.max_loss_risk
        self.min_ev_threshold = self.config.intelligence.min_ev_threshold
        
        logger.info(
            "RealInferenceEngine thresholds: "
            "min_action_confidence=%.2f, max_loss_risk=%.2f, min_ev=%.3f",
            self.min_action_confidence,
            self.max_loss_risk,
            self.min_ev_threshold,
        )
        
        self.system_config: Dict[str, Any] = {}
        self.rl_features: list[str] = []
        self.feature_scaler = None
        self.hmm_model = None
        self.hmm_scaler = None
        self.xgb_loss_model = None
        self.ppo_model = None
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

        self.df: Optional[pd.DataFrame] = None
        self.current_idx: int = 0
        self.total_rows: int = 0

        self.is_ready = False
        self.load_all_artifacts()

    def load_all_artifacts(self):
        try:
            # 1. System Config
            cfg_path = MODELS_DIR / "system_config.json"
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    self.system_config = json.load(f)
                self.rl_features = observation_columns(self.system_config.get("rl_features", []))

            # 2. Feature Scaler
            fs_path = MODELS_DIR / "feature_scaler_v1.pkl"
            if fs_path.exists():
                with open(fs_path, "rb") as f:
                    self.feature_scaler = pickle.load(f)

            # 3. HMM Model & Scaler
            hmm_path = MODELS_DIR / "hmm_v1.pkl"
            hmm_s_path = MODELS_DIR / "hmm_scaler_v1.pkl"
            if hmm_path.exists():
                with open(hmm_path, "rb") as f:
                    self.hmm_model = pickle.load(f)
            if hmm_s_path.exists():
                with open(hmm_s_path, "rb") as f:
                    self.hmm_scaler = pickle.load(f)

            # 4. XGBoost Model
            xgb_path = MODELS_DIR / "xgboost_loss_analyzer_v1.pkl"
            if xgb_path.exists():
                with open(xgb_path, "rb") as f:
                    self.xgb_loss_model = pickle.load(f)

            # 5. PPO Policy
            ppo_path = MODELS_DIR / "ppo_v1.zip"
            if ppo_path.exists():
                try:
                    from sb3_contrib import RecurrentPPO
                    self.ppo_model = RecurrentPPO.load(str(ppo_path), device="cpu")
                except Exception:
                    try:
                        from stable_baselines3 import PPO
                        self.ppo_model = PPO.load(str(ppo_path), device="cpu")
                    except Exception as e:
                        logger.warning(f"Could not load SB3 PPO directly: {e}")

            # 6. Load Parquet Data
            # Prefer 15m or 1m processed dataset
            dataset_candidates = [
                DATA_DIR / "BTCUSDT_15m_features.parquet",
                DATA_DIR / "BTCUSDT_1h_features.parquet",
                DATA_DIR / "BTCUSDT_5m_features.parquet",
                DATA_DIR / "BTCUSDT_multi_timeframe.parquet",
            ]
            for cand in dataset_candidates:
                if cand.exists():
                    logger.info(f"Loading parquet dataset: {cand.name}")
                    self.df = pd.read_parquet(cand)
                    # Sort by timestamp/index
                    if "timestamp" in self.df.columns:
                        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], unit="ms" if pd.api.types.is_numeric_dtype(self.df["timestamp"]) else None)
                        self.df = self.df.sort_values("timestamp").reset_index(drop=True)
                    self.total_rows = len(self.df)
                    # Start from 80% through dataset to simulate recent testing period
                    self.current_idx = max(0, int(self.total_rows * 0.85))
                    break

            self.is_ready = True
            logger.info(f"Real Inference Engine initialized with {self.total_rows} data points.")
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}", exc_info=True)

    def step(self) -> Dict[str, Any]:
        """Execute 1 real inference step through the complete pipeline."""
        if not self.is_ready or self.df is None or self.total_rows == 0:
            return {}

        if self.current_idx >= self.total_rows:
            return {
                "status": "STREAM_COMPLETED",
                "timestamp": time.strftime("%H:%M:%S"),
                "price": float(self.df.iloc[-1].get("close", 0.0)) if self.df is not None and len(self.df) > 0 else 0.0,
                "action_value": 0.0,
                "side": "HOLD",
                "trade_occurred": False,
            }

        # Get current row
        row = self.df.iloc[self.current_idx]
        self.current_idx += 1

        price = float(row.get("close", row.get("close_1m", 64000.0)))
        timestamp_str = str(row.get("timestamp", time.strftime("%H:%M:%S")))

        # Extract features
        obs_features = []
        for feat in self.rl_features:
            if feat in row:
                obs_features.append(float(row[feat]))
            else:
                obs_features.append(0.0)

        obs_array = np.array(obs_features, dtype=np.float32).reshape(1, -1)

        # Scale observation
        if self.feature_scaler is not None:
            try:
                obs_scaled = self.feature_scaler.transform(obs_array)
            except Exception:
                obs_scaled = obs_array
        else:
            obs_scaled = obs_array

        # 1. Real HMM Regime Inference
        regime_name = "UNKNOWN"
        regime_conf = 0.50
        if self.hmm_model is not None and hasattr(self.hmm_model, "predict"):
            try:
                hmm_feats = obs_array[:, :4] if obs_array.shape[1] >= 4 else obs_array
                if self.hmm_scaler is not None:
                    hmm_feats = self.hmm_scaler.transform(hmm_feats)
                state_id = int(self.hmm_model.predict(hmm_feats)[0])
                state_names = ["Bullish Trend", "Bearish Trend", "Mean Reversion", "High Volatility"]
                regime_name = state_names[state_id % len(state_names)]
                if hasattr(self.hmm_model, "predict_proba"):
                    probs = self.hmm_model.predict_proba(hmm_feats)[0]
                    regime_conf = float(probs[state_id])
            except Exception as e:
                logger.warning("HMM inference failure: %s", e)

        # 2. Real PPO Policy Inference (No heuristic fallbacks)
        action_val = 0.0
        if self.ppo_model is not None:
            try:
                if self.lstm_states is not None:
                    action, self.lstm_states = self.ppo_model.predict(
                        obs_scaled,
                        state=self.lstm_states,
                        episode_start=self.episode_starts,
                        deterministic=True,
                    )
                else:
                    action, _ = self.ppo_model.predict(obs_scaled, deterministic=True)
                self.episode_starts = np.zeros((1,), dtype=bool)
                action_val = float(np.clip(action[0], -1.0, 1.0))
            except Exception as e:
                logger.warning("PPO inference failure, defaulting to HOLD (0.0): %s", e)
                action_val = 0.0
        else:
            action_val = 0.0

        # 3. Real XGBoost Loss / Stopout Probability
        loss_risk = 0.12
        if self.xgb_loss_model is not None:
            try:
                if hasattr(self.xgb_loss_model, "predict_proba"):
                    proba = self.xgb_loss_model.predict_proba(obs_array)[0]
                    loss_risk = float(proba[1]) if len(proba) > 1 else float(proba[0])
                else:
                    loss_risk = float(self.xgb_loss_model.predict(obs_array)[0])
            except Exception as e:
                logger.warning("XGBoost loss model inference failure: %s", e)

        # Trade decision logic
        # FIX 18: Use config thresholds instead of magic numbers
        trade_occurred = False
        side = "HOLD"
        quantity = 0.0

        if abs(action_val) > self.min_action_confidence and loss_risk < self.max_loss_risk:
            trade_occurred = True
            side = "BUY" if action_val > 0 else "SELL"
            quantity = round(float(abs(action_val) * 0.8), 2)

        return {
            "timestamp": timestamp_str,
            "price": price,
            "regime": regime_name,
            "regime_confidence": round(regime_conf, 2),
            "action_value": round(action_val, 3),
            "loss_risk": round(loss_risk, 3),
            "trade_occurred": trade_occurred,
            "side": side,
            "quantity": quantity,
            # A signal is not a completed position. Realized PnL is only
            # emitted by execution accounting after an actual exit fill.
            "pnl": 0.0,
            "step_index": self.current_idx,
            "total_steps": self.total_rows,
        }
