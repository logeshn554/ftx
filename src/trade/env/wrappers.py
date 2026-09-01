"""Gymnasium wrappers for observation processing and episode control."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NormalizeObservation(gym.ObservationWrapper):
    """Z-score normalize observations using running statistics.

    Tracks running mean and variance of observations and normalizes
    each new observation to zero mean, unit variance.
    """

    def __init__(self, env: gym.Env, clip: float = 10.0, epsilon: float = 1e-8) -> None:
        super().__init__(env)
        self.clip = clip
        self.epsilon = epsilon

        obs_shape = self.observation_space.shape
        assert obs_shape is not None

        self._running_mean = np.zeros(obs_shape, dtype=np.float64)
        self._running_var = np.ones(obs_shape, dtype=np.float64)
        self._count = 0

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running stats."""
        self._update_stats(obs)
        normalized = (obs - self._running_mean) / np.sqrt(self._running_var + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def _update_stats(self, obs: np.ndarray) -> None:
        """Update running mean and variance with Welford's algorithm."""
        self._count += 1
        delta = obs - self._running_mean
        self._running_mean += delta / self._count
        delta2 = obs - self._running_mean
        self._running_var += (delta * delta2 - self._running_var) / self._count


class FrameStack(gym.ObservationWrapper):
    """Stack N consecutive observations along a new axis.

    Useful when the policy needs to see multiple timesteps at once
    but the base environment only returns a single frame.
    """

    def __init__(self, env: gym.Env, n_frames: int = 4) -> None:
        super().__init__(env)
        self.n_frames = n_frames

        old_space = env.observation_space
        assert isinstance(old_space, spaces.Box)
        assert old_space.shape is not None

        low = np.repeat(old_space.low[np.newaxis, ...], n_frames, axis=0)
        high = np.repeat(old_space.high[np.newaxis, ...], n_frames, axis=0)

        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        self._frames: list[np.ndarray] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames = [obs] * self.n_frames
        return self._get_stacked(), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._frames.append(obs)
        if len(self._frames) > self.n_frames:
            self._frames.pop(0)
        return self._get_stacked()

    def _get_stacked(self) -> np.ndarray:
        return np.array(self._frames, dtype=np.float32)


class MaxDrawdownWrapper(gym.Wrapper):
    """Terminate episode if drawdown exceeds a threshold.

    This wrapper tracks the peak portfolio value and terminates
    the episode early if the drawdown from peak exceeds the limit.
    """

    def __init__(self, env: gym.Env, max_drawdown: float = 0.25) -> None:
        super().__init__(env)
        self.max_drawdown = max_drawdown
        self._peak_value = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._peak_value = info.get("portfolio_value", 0.0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        portfolio_value = info.get("portfolio_value", 0.0)
        self._peak_value = max(self._peak_value, portfolio_value)

        if self._peak_value > 0:
            drawdown = (self._peak_value - portfolio_value) / self._peak_value
            if drawdown > self.max_drawdown:
                terminated = True
                info["termination_reason"] = f"max_drawdown_exceeded ({drawdown:.2%})"

        return obs, reward, terminated, truncated, info
