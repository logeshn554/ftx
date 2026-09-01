"""Frozen-model inference for the Production Agent.

This is the runtime component that loads approved model weights and
produces BUY/SELL/HOLD decisions. It NEVER modifies its own weights.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from trade.core.types import Action

logger = logging.getLogger(__name__)


class ProductionAgent:
    """Production inference agent using frozen (approved) model weights.

    Key guarantees:
        - No gradient computation
        - No weight updates
        - Thread-safe prediction
        - Deterministic output (no exploration noise)
    """

    def __init__(self, model_path: str, device: str = "auto") -> None:
        """Load a frozen model for inference.

        Args:
            model_path: Path to saved SB3 model file.
            device: PyTorch device ("cpu", "cuda", "auto").
        """
        self._model_path = model_path
        self._lock = threading.Lock()
        self._loaded = False
        self._model: PPO | None = None
        self._version: str = ""
        self._device = device

        self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        """Load model weights and freeze them."""
        path_obj = Path(path)

        # SB3 appends .zip if not present
        if not path_obj.suffix:
            if Path(f"{path}.zip").exists():
                path = f"{path}"
            elif path_obj.exists():
                pass
            else:
                raise FileNotFoundError(f"Model not found: {path}")

        self._model = PPO.load(path, device=self._device)

        # Freeze all parameters — belt and suspenders
        for param in self._model.policy.parameters():
            param.requires_grad = False

        self._model.policy.eval()  # Set to evaluation mode
        self._loaded = True
        self._version = path_obj.stem

        logger.info("Production agent loaded: %s (frozen, deterministic)", self._version)

    def predict(self, observation: np.ndarray) -> Action:
        """Produce a trading action from the current observation.

        This method is thread-safe and deterministic.

        Args:
            observation: The current observation from the environment.

        Returns:
            Action enum (HOLD, BUY, or SELL).
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Production agent model not loaded")

        with self._lock:
            action_int, _ = self._model.predict(
                observation,
                deterministic=True,  # No exploration noise in production
            )

        return Action(int(action_int))

    def predict_with_confidence(
        self, observation: np.ndarray
    ) -> tuple[Action, dict[str, float]]:
        """Predict action and return action probabilities.

        Useful for monitoring model confidence and detecting regime changes.

        Returns:
            Tuple of (action, probability_dict) where probability_dict maps
            action names to their probabilities.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Production agent model not loaded")

        import torch

        with self._lock:
            obs_tensor = torch.as_tensor(observation).unsqueeze(0).float()
            obs_tensor = obs_tensor.to(self._model.device)

            with torch.no_grad():
                distribution = self._model.policy.get_distribution(obs_tensor)
                probs = distribution.distribution.probs.cpu().numpy().flatten()

            action_int, _ = self._model.predict(observation, deterministic=True)

        action = Action(int(action_int))
        prob_dict = {
            Action.HOLD.name: float(probs[0]),
            Action.BUY.name: float(probs[1]),
            Action.SELL.name: float(probs[2]),
        }

        return action, prob_dict

    def health_check(self) -> dict[str, bool | str]:
        """Check if the agent is operational.

        Returns:
            Dict with health status fields.
        """
        return {
            "loaded": self._loaded,
            "version": self._version,
            "model_path": self._model_path,
            "device": str(self._model.device) if self._model else "none",
            "frozen": True,
        }

    @property
    def version(self) -> str:
        return self._version

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def swap_model(self, new_model_path: str) -> None:
        """Hot-swap to a new model version (for promotions/rollbacks).

        Thread-safe: acquires lock during swap to prevent concurrent predictions
        on a partially-loaded model.

        Args:
            new_model_path: Path to the new model file.
        """
        with self._lock:
            old_version = self._version
            self._load_model(new_model_path)
            logger.info(
                "Production agent hot-swapped: %s → %s",
                old_version,
                self._version,
            )
