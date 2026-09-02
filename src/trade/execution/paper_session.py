"""Paper trading session using canonical ledger and decision engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from trade.execution.cost_model import CostConfig
from trade.execution.ledger import Ledger
from trade.intelligence.decision_engine import DecisionEngine, TradeDecision


@dataclass
class CloseResult:
    trade_id: str
    close_reason: str
    gross_pnl: float
    net_pnl: float
    fees: float
    slippage: float
    return_pct: float
    tp_price_hit: bool
    sl_price_hit: bool
    time_expired: bool


class PaperTradingSession:
    """Canonical paper-trading runtime for dashboard and scripts."""

    def __init__(self, initial_cash: float = 10_000.0, fee_pct: float = 0.001, slippage_pct: float = 0.0005):
        cost = CostConfig(taker_fee=fee_pct, maker_fee=fee_pct, entry_slippage=slippage_pct, exit_slippage=slippage_pct)
        self.ledger = Ledger(initial_cash, cost)
        self.engine = DecisionEngine(cost_config=cost)
        self._open_meta: dict | None = None
        self._trade_counter = 0

    def map_server_indicators(self, ind: dict, price: float) -> dict:
        """Map dashboard indicator keys to strategy-expected feature keys."""
        ema10 = float(ind.get("ema_10", price))
        ema30 = float(ind.get("ema_30", price))
        rsi = float(ind.get("rsi_14", 50))
        atr_pct = float(ind.get("atr_pct", 0.1))
        momentum = float(ind.get("momentum_20", 0))
        bb_pos = str(ind.get("bb_position", "MID"))
        bb_pct = 0.9 if bb_pos == "ABOVE_UPPER" else 0.1 if bb_pos == "BELOW_LOWER" else 0.5
        return {
            **ind,
            "sma_10": ema10,
            "sma_50": ema30,
            "sma_20": ema30,
            "adx": 30.0 if ind.get("trend") in {"BULLISH", "BEARISH"} else 15.0,
            "rsi_14": rsi,
            "roc_10": momentum / 100.0,
            "momentum_threshold": 0.005,
            "bb_pct": bb_pct,
            "bb_width": float(ind.get("bb_width_pct", 0.01)) / 100.0,
            "atr_pct": atr_pct,
            "close": price,
        }

    def evaluate_entry(
        self,
        indicators: dict,
        price: float,
        regime: str,
        regime_confidence: float,
        drawdown: float = 0.0,
        consecutive_losses: int = 0,
        daily_loss: float = 0.0,
        q_p_win: float | None = None,
        regime_performance: dict[str, dict] | None = None,
    ) -> TradeDecision:
        mapped = self.map_server_indicators(indicators, price)
        return self.engine.decide(
            indicators=mapped,
            equity=self.ledger.equity(price),
            entry_price=price,
            regime=regime,
            regime_confidence=regime_confidence,
            drawdown=drawdown,
            consecutive_losses=consecutive_losses,
            daily_loss=daily_loss,
            p_win=q_p_win,
            regime_performance=regime_performance,
        )

    def open_trade(self, symbol: str, decision: TradeDecision, price: float) -> bool:
        if decision.action != "TRADE" or not decision.side:
            return False
        available = self.ledger.cash
        target_notional = decision.position_size * price
        # For micro-accounts (e.g. $10), allocate 30%-45% of available cash ($3.00 - $4.50)
        # so profit/loss dynamically and visibly updates account balance
        if available <= 50.0:
            target_notional = max(target_notional, min(available * 0.40, 4.50))
        notional = min(available * 0.95, target_notional)
        qty = notional / max(price, 1e-12)
        if qty <= 0:
            return False
        if not self.ledger.enter_position(symbol, decision.side, qty, price):
            return False
        self._trade_counter += 1
        self._open_meta = {
            "trade_id": f"T{self._trade_counter}",
            "tp_pct": decision.target_pct,
            "sl_pct": decision.stop_pct,
            "strategy": decision.strategy,
            "model_version": decision.model_version,
            "entry_time": datetime.now(timezone.utc),
            "duration_bars": 0,
        }
        # Record into anti-churn cooldown tracker
        self.engine.cooldown.record_entry(notional=notional)
        return True

    def check_close(self, price: float, max_bars: int = 6) -> CloseResult | None:
        pos = self.ledger.open_position
        meta = self._open_meta
        if pos is None or meta is None:
            return None
        pos.mark(price)
        meta["duration_bars"] = meta.get("duration_bars", 0) + 1
        tp_pct = float(meta["tp_pct"])
        sl_pct = float(meta["sl_pct"])
        net_pct = pos.return_pct if not pos.is_open else 100.0 * pos.unrealized_pnl() / max(pos.entry_price * pos.quantity, 1e-12)
        gross_pct = 100.0 * pos.gross_pnl / max(pos.entry_price * pos.quantity, 1e-12) if pos.quantity else 0.0

        tp_price_hit = gross_pct >= tp_pct
        sl_price_hit = net_pct <= -abs(sl_pct)
        time_expired = meta["duration_bars"] >= max_bars

        if not (tp_price_hit or sl_price_hit or time_expired):
            return None

        closed = self.ledger.close_position(price)
        if closed is None:
            return None

        if tp_price_hit and closed.net_pnl < 0:
            reason = "TP_PRICE_HIT_NET_LOSS"
        elif tp_price_hit:
            reason = "TP_PRICE_HIT"
        elif sl_price_hit:
            reason = "STOP_LOSS_HIT"
        else:
            reason = "TIME_EXPIRED"

        # Record into anti-churn cooldown tracker
        self.engine.cooldown.record_close(notional=closed.quantity * closed.exit_price)
        self._open_meta = None
        return CloseResult(
            trade_id=meta["trade_id"],
            close_reason=reason,
            gross_pnl=closed.gross_pnl,
            net_pnl=closed.net_pnl,
            fees=closed.entry_fee + closed.exit_fee,
            slippage=closed.slippage_cost,
            return_pct=closed.return_pct,
            tp_price_hit=tp_price_hit,
            sl_price_hit=sl_price_hit,
            time_expired=time_expired,
        )

    def snapshot(self, price: float):
        return self.ledger.snapshot(price)
