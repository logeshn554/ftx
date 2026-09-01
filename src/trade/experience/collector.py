"""Trajectory and reward collector for the experience database.

Attaches to the production agent loop and records full
(state, action, reward, next_state, done, info) tuples for
offline learning.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A single (s, a, r, s', done) transition."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
    timestamp: dt.datetime = field(default_factory=dt.datetime.utcnow)


@dataclass
class Episode:
    """A complete episode of trading transitions."""

    episode_id: str = ""
    symbol: str = ""
    start_time: dt.datetime = field(default_factory=dt.datetime.utcnow)
    end_time: dt.datetime | None = None
    transitions: list[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    total_return: float = 0.0
    regime: str = ""
    model_version: str = ""

    @property
    def length(self) -> int:
        return len(self.transitions)


class ExperienceCollector:
    """Collects trading experience during production execution.

    Attaches to the trading loop and records every state transition.
    Completed episodes are flushed to the ExperienceStore for persistence.
    """

    def __init__(self, model_version: str = "", symbol: str = "") -> None:
        self._model_version = model_version
        self._symbol = symbol
        self._current_episode: Episode | None = None
        self._completed_episodes: list[Episode] = []
        self._episode_counter = 0

    def begin_episode(self, symbol: str = "", regime: str = "") -> str:
        """Start recording a new episode.

        Returns:
            Episode ID.
        """
        self._episode_counter += 1
        episode_id = f"ep_{self._episode_counter}_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        self._current_episode = Episode(
            episode_id=episode_id,
            symbol=symbol or self._symbol,
            model_version=self._model_version,
            regime=regime,
        )

        logger.debug("Episode started: %s", episode_id)
        return episode_id

    def record_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Record a single transition in the current episode."""
        if self._current_episode is None:
            self.begin_episode()

        assert self._current_episode is not None

        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {},
        )

        self._current_episode.transitions.append(transition)
        self._current_episode.total_reward += reward

        if done:
            self.end_episode()

    def end_episode(self) -> Episode | None:
        """Finalize the current episode.

        Returns:
            The completed Episode, or None if no episode was active.
        """
        if self._current_episode is None:
            return None

        ep = self._current_episode
        ep.end_time = dt.datetime.utcnow()

        # Compute total return from info
        if ep.transitions:
            last_info = ep.transitions[-1].info
            ep.total_return = last_info.get("total_return", 0.0)

        self._completed_episodes.append(ep)
        self._current_episode = None

        logger.info(
            "Episode %s completed: %d transitions, reward=%.4f, return=%.4f",
            ep.episode_id,
            ep.length,
            ep.total_reward,
            ep.total_return,
        )

        return ep

    def flush(self) -> list[Episode]:
        """Return and clear all completed episodes.

        Called by the ExperienceStore to persist episodes to the database.
        """
        episodes = self._completed_episodes.copy()
        self._completed_episodes.clear()
        return episodes

    @property
    def current_episode(self) -> Episode | None:
        return self._current_episode

    @property
    def completed_count(self) -> int:
        return len(self._completed_episodes)
