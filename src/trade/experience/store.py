"""Persistent storage for trading experience in the database."""

from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
)
from sqlalchemy.engine import Engine

from trade.experience.collector import Episode, Transition

logger = logging.getLogger(__name__)

exp_metadata = MetaData()

episodes_table = Table(
    "episodes",
    exp_metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("episode_id", String(100), unique=True, nullable=False),
    Column("symbol", String(20)),
    Column("model_version", String(50)),
    Column("regime", String(30)),
    Column("start_time", DateTime),
    Column("end_time", DateTime),
    Column("num_transitions", Integer),
    Column("total_reward", Float),
    Column("total_return", Float),
)

transitions_table = Table(
    "transitions",
    exp_metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("episode_id", String(100), nullable=False, index=True),
    Column("step_index", Integer),
    Column("action", Integer),
    Column("reward", Float),
    Column("done", Integer),  # 0/1
    Column("info_json", Text),
    Column("timestamp", DateTime),
    # States stored as binary blobs for efficiency
    Column("state_blob", LargeBinary),
    Column("next_state_blob", LargeBinary),
)


class ExperienceStore:
    """Persistent storage for trading episodes and transitions.

    Stores episodes and their transitions in the database for
    offline learning and analysis.
    """

    def __init__(self, db_url: str = "sqlite:///trade.db") -> None:
        self._engine: Engine = create_engine(db_url, echo=False)
        exp_metadata.create_all(self._engine)
        logger.info("ExperienceStore initialized")

    def save_episode(self, episode: Episode) -> None:
        """Save a complete episode to the database."""
        with self._engine.begin() as conn:
            # Save episode metadata
            conn.execute(
                episodes_table.insert().values(
                    episode_id=episode.episode_id,
                    symbol=episode.symbol,
                    model_version=episode.model_version,
                    regime=episode.regime,
                    start_time=episode.start_time,
                    end_time=episode.end_time,
                    num_transitions=episode.length,
                    total_reward=episode.total_reward,
                    total_return=episode.total_return,
                )
            )

            # Save transitions
            for i, t in enumerate(episode.transitions):
                conn.execute(
                    transitions_table.insert().values(
                        episode_id=episode.episode_id,
                        step_index=i,
                        action=t.action,
                        reward=t.reward,
                        done=1 if t.done else 0,
                        info_json=json.dumps(
                            {k: v for k, v in t.info.items() if isinstance(v, (int, float, str, bool))}
                        ),
                        timestamp=t.timestamp,
                        state_blob=t.state.tobytes(),
                        next_state_blob=t.next_state.tobytes(),
                    )
                )

        logger.info(
            "Saved episode %s: %d transitions, reward=%.4f",
            episode.episode_id,
            episode.length,
            episode.total_reward,
        )

    def save_episodes(self, episodes: list[Episode]) -> int:
        """Save multiple episodes. Returns count saved."""
        saved = 0
        for ep in episodes:
            try:
                self.save_episode(ep)
                saved += 1
            except Exception:
                logger.warning("Failed to save episode %s", ep.episode_id, exc_info=True)
        return saved

    def load_episodes(
        self,
        symbol: str | None = None,
        model_version: str | None = None,
        regime: str | None = None,
        min_date: dt.date | None = None,
        max_date: dt.date | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Load episode metadata from the database.

        Returns:
            List of episode dicts (without full transition data).
        """
        query = select(episodes_table).order_by(episodes_table.c.start_time.desc()).limit(limit)

        if symbol:
            query = query.where(episodes_table.c.symbol == symbol)
        if model_version:
            query = query.where(episodes_table.c.model_version == model_version)
        if regime:
            query = query.where(episodes_table.c.regime == regime)
        if min_date:
            query = query.where(
                episodes_table.c.start_time >= dt.datetime.combine(min_date, dt.time.min)
            )
        if max_date:
            query = query.where(
                episodes_table.c.end_time <= dt.datetime.combine(max_date, dt.time.max)
            )

        with self._engine.connect() as conn:
            result = conn.execute(query)
            return [dict(row._mapping) for row in result]

    def get_episode_count(self) -> int:
        """Return total number of stored episodes."""
        from sqlalchemy import func

        query = select(func.count()).select_from(episodes_table)
        with self._engine.connect() as conn:
            result = conn.execute(query)
            return result.scalar() or 0

    def get_total_transitions(self) -> int:
        """Return total number of stored transitions."""
        from sqlalchemy import func

        query = select(func.count()).select_from(transitions_table)
        with self._engine.connect() as conn:
            result = conn.execute(query)
            return result.scalar() or 0
