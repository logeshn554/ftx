"""PPO Model Strategy Adapter: wraps trained RL policy into the canonical Strategy interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from trade.data.contract import observation_columns
from trade.strategies.base import Signal, abstain

logger = logging.getLogger(__name__)


class PPOStrategy:
    """Wraps a trained Stable-Baselines3 PPO policy as a specialist Strategy."""

    name = "ppo"

    def __init__(self, model_path: str | Path | None = None, model: Any = None):
        self.model = model
        self.model_path = str(model_path) if model_path else None
        self._load_model()

    def _load_model(self) -> None:
        if self.model is None and self.model_path and Path(self.model_path).exists():
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(self.model_path, device="cpu")
                if hasattr(self.model, "policy"):
                    self.model.policy.eval()
            except Exception as e:
                logger.warning("Could not load PPO model from %s: %s", self.model_path, e)

    def signal(self, indicators: dict) -> Signal:
        """Evaluate indicators through trained PPO policy."""
        if self.model is None:
            return abstain(self.name, "ppo_model_not_loaded")

        # Extract allowed observation columns
        allowed_cols = observation_columns(indicators.keys())
        if not allowed_cols:
            return abstain(self.name, "no_valid_observation_features")

        obs_vec = np.array([float(indicators.get(col, 0.0)) for col in allowed_cols], dtype=np.float32)
        # Reshape to expected (1, window, features) or (1, features)
        if len(obs_vec.shape) == 1:
            obs_vec = obs_vec.reshape(1, -1)

        try:
            action, _ = self.model.predict(obs_vec, deterministic=True)
            act_int = int(action) if np.isscalar(action) or action.ndim == 0 else int(action[0])

            atr_pct = float(indicators.get("atr_pct", indicators.get("atr_14", 1.0)))
            expected_move = max(0.5, atr_pct * 1.5)
            stop_dist = max(0.4, atr_pct * 1.0)
            target_dist = expected_move

            if act_int == 1:  # BUY
                return Signal(
                    side="BUY",
                    confidence=0.75,
                    expected_move=expected_move,
                    stop_distance=stop_dist,
                    target_distance=target_dist,
                    strategy=self.name,
                    reason="ppo_policy_buy_signal",
                )
            elif act_int == 2:  # SELL
                return Signal(
                    side="SELL",
                    confidence=0.75,
                    expected_move=expected_move,
                    stop_distance=stop_dist,
                    target_distance=target_dist,
                    strategy=self.name,
                    reason="ppo_policy_sell_signal",
                )
            else:
                return abstain(self.name, "ppo_policy_hold")
        except Exception as e:
            logger.warning("PPO prediction failed: %s", e)
            return abstain(self.name, f"ppo_inference_error_{e}")
