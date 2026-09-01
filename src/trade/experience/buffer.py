"""In-memory experience replay buffer with prioritized sampling."""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import Any

import numpy as np

from trade.experience.collector import Transition

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """Circular buffer for experience replay.

    Supports:
        - FIFO eviction when capacity is exceeded
        - Uniform random sampling
        - Prioritized sampling by reward magnitude or recency
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._capacity = capacity

    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self._buffer.append(transition)

    def add_batch(self, transitions: list[Transition]) -> None:
        """Add multiple transitions."""
        for t in transitions:
            self._buffer.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        """Uniformly sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of sampled Transitions.
        """
        batch_size = min(batch_size, len(self._buffer))
        return random.sample(list(self._buffer), batch_size)

    def sample_prioritized(
        self,
        batch_size: int,
        by: str = "reward",
        temperature: float = 1.0,
    ) -> list[Transition]:
        """Sample with priority weighting.

        Args:
            batch_size: Number of transitions to sample.
            by: Priority metric — "reward" (magnitude) or "recency".
            temperature: Softmax temperature (higher = more uniform).

        Returns:
            List of prioritized Transitions.
        """
        if len(self._buffer) == 0:
            return []

        batch_size = min(batch_size, len(self._buffer))
        buffer_list = list(self._buffer)

        if by == "reward":
            # Prioritize transitions with larger absolute reward
            priorities = np.array([abs(t.reward) + 1e-6 for t in buffer_list])
        elif by == "recency":
            # Prioritize recent transitions
            priorities = np.array([float(i + 1) for i in range(len(buffer_list))])
        else:
            priorities = np.ones(len(buffer_list))

        # Apply temperature and convert to probabilities
        priorities = priorities ** (1.0 / temperature)
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(buffer_list), size=batch_size, replace=False, p=probs)
        return [buffer_list[i] for i in indices]

    def get_arrays(self) -> dict[str, np.ndarray]:
        """Convert buffer to numpy arrays for batch training.

        Returns:
            Dict with keys: states, actions, rewards, next_states, dones
        """
        if len(self._buffer) == 0:
            return {
                "states": np.array([]),
                "actions": np.array([]),
                "rewards": np.array([]),
                "next_states": np.array([]),
                "dones": np.array([]),
            }

        transitions = list(self._buffer)
        return {
            "states": np.array([t.state for t in transitions]),
            "actions": np.array([t.action for t in transitions]),
            "rewards": np.array([t.reward for t in transitions]),
            "next_states": np.array([t.next_state for t in transitions]),
            "dones": np.array([t.done for t in transitions]),
        }

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self._capacity
