"""PPO training loop using Stable-Baselines3.

Wraps SB3's PPO with project-specific setup: custom feature extractors,
MLflow logging, checkpoint management, and training result packaging.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from trade.core.config import AppConfig, TrainingConfig
from trade.agent.policy import FEATURE_EXTRACTORS
from trade.core.types import ModelVersion, ModelStage, TrainingResult
from trade.env.trading_env import TradingEnv

logger = logging.getLogger(__name__)


class MetricsLoggerCallback(BaseCallback):
    """Custom callback that logs training metrics."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []

    def _on_step(self) -> bool:
        # Collect episode info from Monitor wrapper
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])

                if len(self._episode_rewards) % 10 == 0:
                    recent_rewards = self._episode_rewards[-10:]
                    logger.info(
                        "Episode %d | Avg Reward (last 10): %.4f | Avg Length: %.0f",
                        len(self._episode_rewards),
                        np.mean(recent_rewards),
                        np.mean(self._episode_lengths[-10:]),
                    )
        return True

    def get_summary(self) -> dict[str, float]:
        """Return summary metrics from training."""
        if not self._episode_rewards:
            return {}
        return {
            "total_episodes": len(self._episode_rewards),
            "mean_reward": float(np.mean(self._episode_rewards)),
            "std_reward": float(np.std(self._episode_rewards)),
            "max_reward": float(np.max(self._episode_rewards)),
            "min_reward": float(np.min(self._episode_rewards)),
            "mean_episode_length": float(np.mean(self._episode_lengths)),
            "final_10_mean_reward": float(np.mean(self._episode_rewards[-10:])),
        }


class AgentTrainer:
    """Manages PPO training with project conventions.

    Handles:
        - Creating the PPO model with custom feature extractor
        - Setting up training callbacks (eval, checkpoint, metrics)
        - Running training and packaging results
        - Saving versioned checkpoints
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.tc = config.training
        self._model: PPO | None = None

    def create_model(
        self,
        train_env: TradingEnv,
        eval_env: TradingEnv | None = None,
    ) -> PPO:
        """Create a new PPO model with configured architecture.

        Args:
            train_env: Training environment (will be wrapped with Monitor).
            eval_env: Optional evaluation environment.

        Returns:
            Configured but untrained PPO model.
        """
        # Wrap with Monitor for episode stats
        monitored_env = Monitor(train_env)

        # Select feature extractor
        extractor_class = FEATURE_EXTRACTORS.get(self.tc.policy_architecture)
        if extractor_class is None:
            logger.warning(
                "Unknown architecture '%s', falling back to MLP",
                self.tc.policy_architecture,
            )
            extractor_class = FEATURE_EXTRACTORS["mlp"]

        policy_kwargs = {
            "features_extractor_class": extractor_class,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": {
                "pi": self.tc.net_arch_pi,
                "vf": self.tc.net_arch_vf,
            },
        }

        model = PPO(
            "MlpPolicy",
            monitored_env,
            learning_rate=self.tc.learning_rate,
            n_steps=self.tc.n_steps,
            batch_size=self.tc.batch_size,
            n_epochs=self.tc.n_epochs,
            gamma=self.tc.gamma,
            gae_lambda=self.tc.gae_lambda,
            clip_range=self.tc.clip_range,
            ent_coef=self.tc.ent_coef,
            vf_coef=self.tc.vf_coef,
            max_grad_norm=self.tc.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="auto",
        )

        self._model = model
        logger.info(
            "Created PPO model with %s extractor, %d total params",
            self.tc.policy_architecture,
            sum(p.numel() for p in model.policy.parameters()),
        )
        return model

    def train(
        self,
        train_env: TradingEnv,
        eval_env: TradingEnv | None = None,
        model_version: ModelVersion | None = None,
        checkpoint_dir: str = "checkpoints",
    ) -> TrainingResult:
        """Run the full training loop.

        Args:
            train_env: Training environment.
            eval_env: Evaluation environment for periodic assessment.
            model_version: Version tag for the resulting model.
            checkpoint_dir: Directory to save intermediate checkpoints.

        Returns:
            TrainingResult with metrics and checkpoint path.
        """
        if self._model is None:
            self.create_model(train_env, eval_env)

        assert self._model is not None

        # Setup callbacks
        callbacks: list[BaseCallback] = []

        # Metrics logger
        metrics_cb = MetricsLoggerCallback()
        callbacks.append(metrics_cb)

        # Checkpoints
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_cb = CheckpointCallback(
            save_freq=self.tc.checkpoint_freq,
            save_path=str(ckpt_dir),
            name_prefix="ppo_trading",
        )
        callbacks.append(ckpt_cb)

        # Evaluation
        if eval_env is not None:
            eval_monitored = Monitor(eval_env)
            eval_cb = EvalCallback(
                eval_monitored,
                best_model_save_path=str(ckpt_dir / "best"),
                log_path=str(ckpt_dir / "eval_logs"),
                eval_freq=self.tc.eval_freq,
                n_eval_episodes=5,
                deterministic=True,
            )
            callbacks.append(eval_cb)

        # Train
        logger.info(
            "Starting training: %d timesteps, lr=%.1e, architecture=%s",
            self.tc.total_timesteps,
            self.tc.learning_rate,
            self.tc.policy_architecture,
        )
        start_time = time.time()

        self._model.learn(
            total_timesteps=self.tc.total_timesteps,
            callback=callbacks,
            # Avoid making the core training path depend on optional UI extras.
            progress_bar=False,
        )

        training_time = time.time() - start_time

        # Save final model
        if model_version is None:
            model_version = ModelVersion(major=0, minor=1, patch=0)

        final_path = str(ckpt_dir / f"model_{model_version.tag}")
        self._model.save(final_path)

        # Package results
        metrics = metrics_cb.get_summary()
        result = TrainingResult(
            model_version=model_version,
            total_timesteps=self.tc.total_timesteps,
            training_time_seconds=training_time,
            final_metrics=metrics,
            checkpoint_path=final_path,
        )

        logger.info(
            "Training complete in %.1fs | %d episodes | Mean reward: %.4f",
            training_time,
            metrics.get("total_episodes", 0),
            metrics.get("mean_reward", 0),
        )

        return result

    def load_model(self, path: str, env: TradingEnv | None = None) -> PPO:
        """Load a previously saved model.

        Args:
            path: Path to the saved model (without .zip extension).
            env: Optional environment to attach.

        Returns:
            Loaded PPO model.
        """
        monitored = Monitor(env) if env else None
        self._model = PPO.load(path, env=monitored, device="auto")
        logger.info("Loaded model from %s", path)
        return self._model

    @property
    def model(self) -> PPO | None:
        """Access the current model."""
        return self._model
