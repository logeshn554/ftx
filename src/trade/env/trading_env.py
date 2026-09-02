"""Causal Gymnasium environment with explicit portfolio accounting."""
from __future__ import annotations

from typing import Any
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from trade.core.types import Action
from trade.execution.accounting import PositionAccounting
from trade.env.rewards import portfolio_reward


class TradingEnv(gym.Env):
    """Single-asset long/short environment with cash + signed-position equity."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, features_df: pd.DataFrame | None = None, feature_columns: list[str] | None = None, initial_capital: float = 100_000.0,
                 commission_pct: float = 0.001, slippage_pct: float = 0.0005, position_size_pct: float = 10.0,
                 feature_window: int = 30, reward_function: str = "risk_adjusted", max_drawdown_limit: float = 0.25,
                 reward_drawdown_penalty: float = 0.5, reward_turnover_penalty: float = 0.05,
                 reward_risk_penalty: float = 0.0, render_mode: str | None = None, *, df: pd.DataFrame | None = None,
                 features: pd.DataFrame | None = None, config=None) -> None:
        super().__init__()
        features_df = features_df if features_df is not None else features
        if features_df is None:
            features_df = df
        if features_df is None:
            raise ValueError("features_df (or legacy features/df) is required")
        if config is not None:
            initial_capital = config.trading.initial_capital
            commission_pct = config.trading.commission_pct
            slippage_pct = config.trading.slippage_pct
            position_size_pct = config.trading.default_position_size_pct
            feature_window = config.data.feature_window
            reward_function = config.training.reward_function
            reward_drawdown_penalty = config.training.reward_drawdown_penalty
            reward_turnover_penalty = config.training.reward_turnover_penalty
            reward_risk_penalty = config.training.reward_risk_penalty
        if feature_columns is None:
            from trade.data.contract import OBSERVATION_FEATURES
            feature_columns = [c for c in OBSERVATION_FEATURES if c in features_df.columns]
        self.features_df = features_df.reset_index(drop=True).copy()
        self.feature_columns = list(feature_columns)
        forbidden = [c for c in self.feature_columns if c not in self.features_df or _is_non_observation_column(c)]
        if forbidden:
            raise ValueError(f"feature_columns contain non-observation columns: {forbidden}")
        self.initial_capital, self.commission_pct, self.slippage_pct = initial_capital, commission_pct, slippage_pct
        self.position_size_pct, self.feature_window = position_size_pct / 100.0, feature_window
        self.reward_function, self.max_drawdown_limit = reward_function, max_drawdown_limit
        self.reward_drawdown_penalty, self.reward_turnover_penalty, self.reward_risk_penalty = reward_drawdown_penalty, reward_turnover_penalty, reward_risk_penalty
        self.n_features, self.n_portfolio_features = len(self.feature_columns), 6
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, (feature_window, self.n_features + self.n_portfolio_features), dtype=np.float32)
        self.reset()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._current_step = self.feature_window
        self._accounting = PositionAccounting(self.initial_capital, self.commission_pct, self.slippage_pct)
        self._portfolio_value = self.initial_capital
        self._max_portfolio_value = self.initial_capital
        self._trade_log: list[dict[str, Any]] = []
        self._returns_history: list[float] = []
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        previous_equity, previous_drawdown, prior_turnover = self._portfolio_value, self._get_drawdown(), self._accounting.turnover
        price = self._get_current_price()
        action_enum = Action(action)
        self._execute_action(action_enum, price)
        self._accounting.advance()
        self._portfolio_value = self._accounting.equity(price)
        self._max_portfolio_value = max(self._max_portfolio_value, self._portfolio_value)
        step_return = (self._portfolio_value - previous_equity) / max(previous_equity, 1e-12)
        self._returns_history.append(step_return)
        reward = portfolio_reward(step_return, self._get_drawdown() - previous_drawdown, self._accounting.turnover - prior_turnover, self._portfolio_value, self.reward_drawdown_penalty, self.reward_turnover_penalty, self.reward_risk_penalty)
        self._current_step += 1
        terminated = self._portfolio_value <= self.initial_capital * 0.5 or self._get_drawdown() > self.max_drawdown_limit
        truncated = self._current_step >= len(self.features_df)
        if truncated and self._accounting.position:
            self.force_flat(reason="LIQUIDATE")
        info = self._get_info()
        info.update(action=action_enum.name, step_return=step_return)
        return self._get_observation(), reward, terminated, truncated, info

    def force_flat(self, price: float | None = None, reason: str = "LIQUIDATE") -> Any:
        """Force position closure to flat outside of policy action space (e.g. risk/stop/kill switch)."""
        current_price = price if price is not None else self._get_current_price()
        if self._accounting.position:
            closed = self._accounting.close(current_price)
            if closed:
                self._trade_log.append({"step": self._current_step, "action": reason, **closed.to_dict()})
                self._portfolio_value = self._accounting.equity(current_price)
                return closed
        return None

    def _execute_action(self, action: Action, price: float) -> None:
        # Target Position Semantics:
        # Action.HOLD (0) -> Maintain current exposure (zero turnover, zero fees).
        # Action.BUY (1)  -> Target Long (+1). Holds if already long, opens if flat, flips short->long on same step.
        # Action.SELL (2) -> Target Short (-1). Holds if already short, opens if flat, flips long->short on same step.
        if action == Action.HOLD:
            return

        target_side = "BUY" if action == Action.BUY else "SELL"
        if self._accounting.side == target_side:
            # Already in desired exposure: hold position with zero turnover & zero fees
            return

        if self._accounting.position != 0:
            # Reversal: close opposite exposure to flat before opening target exposure on same step
            closed = self._accounting.close(price)
            if closed:
                self._trade_log.append({"step": self._current_step, "action": "CLOSE", **closed.to_dict()})

        # Open new position
        quantity = max(0.0, self._accounting.equity(price) * self.position_size_pct / max(price, 1e-12))
        if self._accounting.open(target_side, quantity, price):
            self._trade_log.append({
                "step": self._current_step,
                "action": target_side,
                "entry_price": self._accounting.entry_price,
                "quantity": quantity,
                "side": target_side,
                "entry_fee": self._accounting.entry_fee,
            })

    def _get_observation(self) -> np.ndarray:
        start, end = max(0, self._current_step - self.feature_window + 1), self._current_step + 1
        technical = self.features_df.iloc[start:end][self.feature_columns].to_numpy()
        if len(technical) < self.feature_window:
            technical = np.vstack([np.zeros((self.feature_window - len(technical), self.n_features)), technical])
        price = self._get_current_price()
        portfolio = np.array([self._accounting.cash / self.initial_capital, self._accounting.position * price / max(self._portfolio_value, 1e-12), self._accounting.unrealized_pnl(price) / self.initial_capital, self._get_drawdown(), self._accounting.realized_pnl / self.initial_capital, self._accounting.turnover / self.initial_capital], dtype=np.float32)
        return np.nan_to_num(np.hstack([technical, np.tile(portfolio, (self.feature_window, 1))]), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    def _get_current_price(self) -> float:
        return float(self.features_df.iloc[min(self._current_step, len(self.features_df) - 1)]["close"])

    def _get_drawdown(self) -> float:
        return max(0.0, (self._max_portfolio_value - self._portfolio_value) / max(self._max_portfolio_value, 1e-12))

    def _get_info(self) -> dict[str, Any]:
        price = self._get_current_price()
        return {"step": self._current_step, "price": price, "cash": self._accounting.cash, "equity": self._portfolio_value, "portfolio_value": self._portfolio_value, "position": self._accounting.position, "position_side": self._accounting.side, "entry_price": self._accounting.entry_price, "quantity": self._accounting.quantity, "unrealized_pnl": self._accounting.unrealized_pnl(price), "realized_pnl": self._accounting.realized_pnl, "fees": self._accounting.total_fees, "slippage_cost": self._accounting.total_slippage_cost, "turnover": self._accounting.turnover, "max_portfolio_value": self._max_portfolio_value, "peak_equity": self._max_portfolio_value, "drawdown": self._get_drawdown(), "total_commission": self._accounting.total_fees, "total_trades": len([t for t in self._trade_log if "net_pnl" in t]), "total_return": (self._portfolio_value - self.initial_capital) / self.initial_capital}

    @property
    def trade_log(self) -> list[dict[str, Any]]:
        return self._trade_log.copy()


def _is_non_observation_column(column: str) -> bool:
    name = column.lower()
    return "future" in name or "target" in name or name.startswith("return_1_")
