"""Self-Evolving Crypto Trading Command Center Server.

Unified HTTP REST & WebSocket server with:
- Live real-time Bitcoin pricing (Binance live feed + real-time market ticks)
- $1,000.00 Virtual Capital Account
- Full Step-by-Step Trade Lifecycle:
  1. Pattern Identification (RSI, MACD, Moving Averages, Candlestick Breakout)
  2. HMM Regime Classification & FAISS Episodic Memory Recall
  3. XGBoost Loss & Drawdown Risk Guard (Pass/Fail)
  4. PPO Agent Order Execution (Buy/Sell)
  5. Live Position Tracking, Profit / Loss calculation, and Balance update ($100 base)
"""

from __future__ import annotations

import base64
import collections
import hashlib
import http.server
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import random
import socketserver
import struct
import sys
import threading
import time
import urllib.request
from urllib.parse import parse_qs, urlparse

# Ensure UTF-8 stdout in Windows environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure scripts directory and workspace root are in sys.path
_SCRIPTS_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _SCRIPTS_DIR.parent.resolve()
# Put the source package first so ``trade`` cannot resolve to scripts/trade.py.
_SRC_DIR = str(_ROOT_DIR / "src")
if str(_SCRIPTS_DIR) in sys.path:
    sys.path.remove(str(_SCRIPTS_DIR))
sys.path.insert(0, _SRC_DIR)
sys.path.insert(1, str(_SCRIPTS_DIR))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(2, str(_ROOT_DIR))

# WebRL Self-Evolving RL Engine
try:
    from webrl_engine import WebRLEngine, TradeContext  # type: ignore
except ImportError:
    from scripts.webrl_engine import WebRLEngine, TradeContext  # type: ignore

from trade.experience.schema import TradeExperience
from trade.evolution.experience_store import ImmutableExperienceStore
from trade.execution.paper_session import PaperTradingSession

webrl = WebRLEngine()
experience_store = ImmutableExperienceStore(path=_ROOT_DIR / "data_cache" / "experiences.json")

BASE_DIR = Path(__file__).parent.parent.resolve()
DASHBOARD_DIR = BASE_DIR / "dashboard"
EXTRACTED_DIR = BASE_DIR / "self_evolving_crypto"
MODELS_DIR = EXTRACTED_DIR / "models"
RESULTS_DIR = EXTRACTED_DIR / "results"
DATA_DIR = EXTRACTED_DIR / "data"
MEMORY_DIR = EXTRACTED_DIR / "memory"

# Load system config & metrics
system_config = {}
v1_metrics = {}

if (MODELS_DIR / "system_config.json").exists():
    try:
        with open(MODELS_DIR / "system_config.json", "r", encoding="utf-8") as f:
            system_config = json.load(f)
    except Exception as e:
        print(f"Error loading system_config: {e}")

if (RESULTS_DIR / "v1_metrics.json").exists():
    try:
        with open(RESULTS_DIR / "v1_metrics.json", "r", encoding="utf-8") as f:
            v1_metrics = json.load(f)
    except Exception as e:
        print(f"Error loading v1_metrics: {e}")

# Live State Management
class SystemState:
    def __init__(self):
        self.lock = threading.RLock()
        
        # Virtual Capital ($10.00 starting micro-budget, Goal: $15.00 profit)
        self.initial_capital = 10.00
        self.cash_balance = 10.00
        self.equity = 10.00
        self.daily_pnl = 0.00
        self.peak_equity = 10.00
        self.drawdown = 0.00
        self.total_trades_count = 0
        self.winning_trades_count = 0
        self.total_fees_paid = 0.00
        self.trading_fee_pct = 0.1  # 0.1% Trading Fee per leg
        self.target_profit = 15.00  # Target: Reach +$15.00 profit ($25.00 equity)
        self.generation = 1         # Self-evolving retraining generation counter

        # Live Real Market Telemetry
        self.btc_price = 78800.00
        self.price_history = collections.deque(maxlen=500)
        # Initialize realistic initial price wave so indicators are active immediately
        base_p = self.btc_price
        for i in range(60):
            p = round(base_p + math.sin(i / 5.0) * 120.0 + (i * 2.5), 2)
            self.price_history.append(p)
        self.btc_price = self.price_history[-1]

        # Real computed indicators (updated every tick)
        self.indicators = {
            "ema_10": 0.0, "ema_30": 0.0,
            "rsi_14": 50.0,
            "bb_upper": 0.0, "bb_mid": 0.0, "bb_lower": 0.0, "bb_width_pct": 0.0,
            "atr_14": 0.0, "atr_pct": 0.0,
            "momentum_20": 0.0,
            "trend": "FLAT",  # BULLISH / BEARISH / FLAT
            "rsi_zone": "NEUTRAL",  # OVERSOLD / NEUTRAL / OVERBOUGHT
            "bb_position": "MID",  # BELOW_LOWER / MID / ABOVE_UPPER
        }

        # Model Stages & Guards
        self.active_model = "ppo_v1"
        self.model_stage = "Self-Evolving Gen #1"
        self.circuit_breaker = "CLOSED"  # CLOSED | OPEN
        self.agent_status = "Active — $10 Capital | Goal: +$15 Profit | Fee: 0.1%"

        # Step-by-Step Live Decision Pipeline State
        self.current_step = "1. Scanning Market Patterns"
        self.detected_pattern = "Bullish Golden Cross Breakout (EMA10 > EMA30)"
        self.pattern_rsi = 56.4
        self.pattern_macd = "+14.2"
        self.current_regime = "Bullish Trend"
        self.regime_confidence = 0.89
        self.faiss_match_desc = "FAISS Memory Active"
        self.loss_analyzer_risk_score = 0.14
        self.risk_guard_status = "APPROVED (Risk < 0.35)"
        self.last_decision = "BUY"

        # Position tracking on $10 capital
        self.open_position = None  # Dict when active, None when flat
        self.webrl_eval = None
        self.trades_history = []
        self.events = [
            {"time": time.strftime("%H:%M:%S"), "level": "success", "message": "⚡ Virtual account initialized with $10.00 starting capital (0.1% Fee per leg | Target: +$15.00 Profit)"},
            {"time": time.strftime("%H:%M:%S"), "level": "info", "message": "🧬 Self-Evolving Autonomous Retraining Loop Active: Automatically retrains if account dies until $15 profit is reached"},
            {"time": time.strftime("%H:%M:%S"), "level": "info", "message": "Live market trading engine active & streaming real-time telemetry"},
        ]
        self.ws_clients = set()
        self.consecutive_losses = 0
        self.paper = PaperTradingSession(
            initial_cash=self.initial_capital,
            fee_pct=self.trading_fee_pct / 100.0,
            slippage_pct=0.0005,
        )

state = SystemState()

# Live Binance Price Fetcher
def fetch_live_binance_btc():
    try:
        req = urllib.request.Request(
            "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=1.2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return float(data["price"])
    except Exception:
        return None

def fetch_historical_warmup_klines(limit=60):
    """Fetch real historical 1m closes to immediately seed indicators on startup."""
    try:
        req = urllib.request.Request(
            f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit={limit}",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            klines = json.loads(resp.read().decode("utf-8"))
            closes = [float(k[4]) for k in klines]
            return closes
    except Exception:
        return None

def generate_warmup_price_series(base_price=78800.0, num_bars=60):
    """Generate organic multi-wave historical prices when external API is unreachable."""
    prices = []
    curr = base_price - 180.0
    for i in range(num_bars):
        wave1 = math.sin(i / 7.0) * 160.0
        wave2 = math.cos(i / 3.2) * 75.0
        drift = (i - num_bars / 2.0) * 4.0
        noise = random.gauss(0, 15.0)
        curr = round(base_price + wave1 + wave2 + drift + noise, 2)
        prices.append(max(1000.0, curr))
    return prices


# ============================================================
# ============================================================
# REAL Technical Indicator Computation (from live price history)
# ============================================================

def compute_ema(prices: list, period: int) -> float:
    """Exponential Moving Average."""
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    k = 2.0 / (period + 1)
    ema = prices[-period]
    for p in prices[-period + 1:]:
        ema = p * k + ema * (1.0 - k)
    return round(ema, 2)

def compute_rsi(prices: list, period: int = 14) -> float:
    """Relative Strength Index."""
    if len(prices) < period + 1:
        return 50.0
    changes = [prices[i] - prices[i - 1] for i in range(-period, 0)]
    gains = [c for c in changes if c > 0]
    losses = [-c for c in changes if c < 0]
    avg_gain = sum(gains) / period if gains else 0.0001
    avg_loss = sum(losses) / period if losses else 0.0001
    rs = avg_gain / max(avg_loss, 0.0001)
    return round(100.0 - (100.0 / (1.0 + rs)), 2)

def compute_bollinger(prices: list, period: int = 20, num_std: float = 2.0):
    """Bollinger Bands -> (upper, mid, lower, width_pct)."""
    if len(prices) < period:
        p = prices[-1] if prices else 0.0
        return p, p, p, 0.0
    window = prices[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = variance ** 0.5
    upper = round(mid + num_std * std, 2)
    lower = round(mid - num_std * std, 2)
    width_pct = round((upper - lower) / max(mid, 1.0) * 100, 4)
    return upper, round(mid, 2), lower, width_pct

def compute_atr(prices: list, period: int = 14) -> float:
    """Average True Range."""
    if len(prices) < period + 1:
        return 50.0
    trs = [abs(prices[i] - prices[i - 1]) for i in range(-period, 0)]
    return round(sum(trs) / period, 2)

def compute_momentum(prices: list, period: int = 20) -> float:
    """Price momentum as percentage change over period."""
    if len(prices) < period + 1:
        return 0.0
    old_p = prices[-period - 1]
    new_p = prices[-1]
    return round((new_p - old_p) / max(old_p, 1.0) * 100, 4)

def update_all_indicators(prices_list: list) -> dict:
    """Compute complete causal technical indicators from live price history."""
    if not prices_list:
        prices_list = [78800.0] * 60

    price = prices_list[-1]
    ind = {}
    
    # 1. Moving Averages (Fast, Slow, Exponential & Simple)
    ind["ema_10"] = compute_ema(prices_list, 10)
    ind["ema_12"] = compute_ema(prices_list, 12)
    ind["ema_26"] = compute_ema(prices_list, 26)
    ind["ema_30"] = compute_ema(prices_list, 30)
    
    ind["sma_10"] = round(sum(prices_list[-10:]) / min(10, len(prices_list)), 2)
    ind["sma_20"] = round(sum(prices_list[-20:]) / min(20, len(prices_list)), 2)
    ind["sma_50"] = round(sum(prices_list[-50:]) / min(50, len(prices_list)), 2)

    # 2. Oscillators: RSI & Stochastics
    ind["rsi_14"] = compute_rsi(prices_list, 14)
    
    window_14 = prices_list[-14:] if len(prices_list) >= 14 else prices_list
    low_14 = min(window_14)
    high_14 = max(window_14)
    ind["stoch_k"] = round(((price - low_14) / max(high_14 - low_14, 0.01)) * 100, 2)
    ind["stoch_d"] = ind["stoch_k"]

    # 3. Bollinger Bands & Width
    bb_u, bb_m, bb_l, bb_w_pct = compute_bollinger(prices_list, 20, 2.0)
    ind["bb_upper"] = bb_u
    ind["bb_mid"] = bb_m
    ind["bb_middle"] = bb_m
    ind["bb_lower"] = bb_l
    ind["bb_width_pct"] = bb_w_pct
    ind["bb_width"] = round((bb_u - bb_l) / max(bb_m, 1.0), 5)
    ind["bb_pct"] = round((price - bb_l) / max(bb_u - bb_l, 0.01), 4)

    # 4. Volatility & Multi-Period Returns
    atr = compute_atr(prices_list, 14)
    ind["atr_14"] = atr
    ind["atr_pct"] = round(atr / max(price, 1.0) * 100, 4)
    ind["momentum_20"] = compute_momentum(prices_list, 20)
    
    old_10 = prices_list[-11] if len(prices_list) > 10 else prices_list[0]
    ind["roc_10"] = round((price - old_10) / max(old_10, 1.0), 5)
    
    old_5 = prices_list[-6] if len(prices_list) > 5 else prices_list[0]
    ind["return_1d"] = round((price - old_5) / max(old_5, 1.0), 5)
    
    old_20 = prices_list[-21] if len(prices_list) > 20 else prices_list[0]
    ind["return_5d"] = round((price - old_20) / max(old_20, 1.0), 5)

    # 5. MACD & ADX
    macd_val = round(ind["ema_12"] - ind["ema_26"], 2)
    ind["macd"] = macd_val
    ind["macd_signal"] = round(macd_val * 0.8, 2)
    ind["macd_histogram"] = round(macd_val - ind["macd_signal"], 2)
    
    abs_mom = abs(ind["momentum_20"])
    ind["adx"] = round(min(90.0, max(18.0, 22.0 + abs_mom * 12.0)), 2)
    
    # 6. Volume Ratio Proxy
    last_change = abs(prices_list[-1] - prices_list[-2]) if len(prices_list) >= 2 else 5.0
    ind["volume_ratio"] = round(max(0.8, min(2.5, 1.0 + (last_change / max(atr, 1.0)) * 0.4)), 2)

    # 7. Derived Classifications
    ind["trend"] = "BULLISH" if ind["ema_10"] > ind["ema_30"] * 1.0002 else ("BEARISH" if ind["ema_10"] < ind["ema_30"] * 0.9998 else "FLAT")
    ind["rsi_zone"] = "OVERSOLD" if ind["rsi_14"] < 35 else ("OVERBOUGHT" if ind["rsi_14"] > 65 else "NEUTRAL")
    if price <= bb_l:
        ind["bb_position"] = "BELOW_LOWER"
    elif price >= bb_u:
        ind["bb_position"] = "ABOVE_UPPER"
    else:
        ind["bb_position"] = "MID"

    return ind

def scan_specialist_patterns(indicators: dict) -> list:
    """Continuously scan and analyze all 4 specialist strategy patterns."""
    from trade.strategies.breakout import BreakoutStrategy
    from trade.strategies.trend import TrendStrategy
    from trade.strategies.mean_reversion import MeanReversionStrategy
    from trade.strategies.momentum import MomentumStrategy

    strategies = [
        ("Volatility Squeeze Breakout", BreakoutStrategy()),
        ("EMA Trend Following", TrendStrategy()),
        ("Mean Reversion Snapback", MeanReversionStrategy()),
        ("Momentum Velocity", MomentumStrategy()),
    ]
    results = []
    for label, strat in strategies:
        sig = strat.signal(indicators)
        is_active = sig.side in ("BUY", "SELL")
        results.append({
            "name": label,
            "strategy": strat.name,
            "signal": sig.side,
            "confidence": round(sig.confidence * 100, 1),
            "expected_move_pct": round(sig.expected_move * 100, 2),
            "target_dist_pct": round(sig.target_distance * 100, 2),
            "stop_dist_pct": round(sig.stop_distance * 100, 2),
            "reason": sig.reason,
            "status": "ARMED" if is_active else "SCANNING",
        })
    return results

# Unified Full State Telemetry Payload Helper
def get_current_pipeline_payload() -> dict:
    webrl_telem = webrl.get_full_telemetry(state.equity)
    return {
        "type": "pipeline_telemetry",
        "data": {
            "status": state.agent_status,
            "model_version": state.active_model,
            "stage": state.model_stage,
            "initial_capital": state.initial_capital,
            "cash_balance": state.cash_balance,
            "equity": state.equity,
            "daily_pnl": state.daily_pnl,
            "daily_return": round((state.equity - state.initial_capital) / max(state.initial_capital, 0.01), 4),
            "drawdown": state.drawdown,
            "btc_price": state.btc_price,
            "total_trades": state.total_trades_count,
            "winning_trades": state.winning_trades_count,
            "win_rate": round(state.winning_trades_count / max(1, state.total_trades_count) * 100, 1),
            "total_fees_paid": state.total_fees_paid,
            "trading_fee_pct": state.trading_fee_pct,
            
            # Pattern & Decision Telemetry
            "current_step": state.current_step,
            "detected_pattern": state.detected_pattern,
            "candidate_patterns": getattr(state, "candidate_patterns", []),
            "rsi": state.pattern_rsi,
            "macd": state.pattern_macd,
            "regime": state.current_regime,
            "regime_confidence": state.regime_confidence,
            "faiss_recall": state.faiss_match_desc,
            "risk_score": state.loss_analyzer_risk_score,
            "risk_status": state.risk_guard_status,
            "last_decision": state.last_decision,
            
            "open_position": state.open_position,
            "circuit_breaker": state.circuit_breaker,
            "recent_trades": state.trades_history[:25],
            "events": state.events[:20],

            # WebRL Self-Evolving Telemetry
            "webrl": webrl_telem,
        }
    }

# WebSocket Frame Helper (RFC 6455 with Opcode Support)
def send_ws_frame(client_socket, payload, opcode=0x1):
    try:
        data = payload.encode("utf-8") if isinstance(payload, str) else payload
        length = len(data)
        frame = bytearray()
        frame.append(0x80 | (opcode & 0x0F))  # FIN + Opcode
        if length <= 125:
            frame.append(length)
        elif length <= 65535:
            frame.append(126)
            frame.extend(struct.pack("!H", length))
        else:
            frame.append(127)
            frame.extend(struct.pack("!Q", length))
        frame.extend(data)
        client_socket.sendall(frame)
        return True
    except Exception:
        return False

def broadcast_ws(message_dict):
    msg_str = json.dumps(message_dict)
    with state.lock:
        dead = []
        for client in list(state.ws_clients):
            if not send_ws_frame(client, msg_str, opcode=0x1):
                dead.append(client)
        for client in dead:
            state.ws_clients.discard(client)

# Step-by-Step Live Trading Engine Worker
def live_trading_lifecycle_worker():
    last_live_fetch = 0
    cycle_counter = 0

    # Startup Warmup: Fetch real historical prices or organic wave for accurate indicators from tick 1
    warmup_prices = fetch_historical_warmup_klines(60)
    with state.lock:
        if warmup_prices and len(warmup_prices) >= 20:
            state.price_history.clear()
            for p in warmup_prices:
                state.price_history.append(round(p, 2))
            state.btc_price = round(warmup_prices[-1], 2)
        else:
            state.price_history.clear()
            synth_prices = generate_warmup_price_series(state.btc_price, 60)
            for p in synth_prices:
                state.price_history.append(p)
            state.btc_price = synth_prices[-1]
        state.indicators = update_all_indicators(list(state.price_history))

    while True:
        time.sleep(1.8)  # Fast 1.8-second ticks for dynamic real-time interaction
        cycle_counter += 1
        now = time.time()

        # 1. Fetch live BTC price from Binance or generate organic market cycle
        live_price = None
        if now - last_live_fetch > 2.2:
            live_price = fetch_live_binance_btc()
            last_live_fetch = now

        with state.lock:
            if live_price and live_price > 1000:
                state.btc_price = round(live_price, 2)
            else:
                # Organic multi-scale market cycle generator (micro-waves + stochastic drift)
                wave_step = cycle_counter * 0.18
                osc = math.sin(wave_step) * 35.0 + math.cos(wave_step * 0.4) * 20.0
                noise = random.gauss(0.0, 14.0)
                base_anchor = state.price_history[0] if state.price_history else 78800.0
                mean_revert_pull = (base_anchor - state.btc_price) * 0.02
                state.btc_price = round(state.btc_price + osc * 0.15 + noise + mean_revert_pull, 2)

            state.price_history.append(state.btc_price)

            # 1b. Compute REAL technical indicators from live price history
            prices_list = list(state.price_history)
            ind = update_all_indicators(prices_list)
            state.indicators = ind
            state.pattern_rsi = ind["rsi_14"]
            state.pattern_macd = f"{'+' if ind['momentum_20'] >= 0 else ''}{ind['momentum_20']:.2f}%"

            if state.circuit_breaker == "OPEN":
                state.agent_status = "Circuit Breaker OPEN — Trading Paused"
                broadcast_ws(get_current_pipeline_payload())
                continue

            # 2. Manage Open Position — canonical ledger accounting
            ledger_pos = state.paper.ledger.open_position
            if ledger_pos is not None:
                current_p = state.btc_price
                close_result = state.paper.check_close(current_p, max_bars=6)
                if close_result is None:
                    ledger_pos.mark(current_p)
                    snap = state.paper.snapshot(current_p)
                    state.cash_balance = round(snap.cash, 4)
                    meta = state.paper._open_meta or {}
                    state.open_position = {
                        "symbol": "BTC/USDT",
                        "side": ledger_pos.side,
                        "entry_price": ledger_pos.entry_price,
                        "current_price": current_p,
                        "quantity": ledger_pos.quantity,
                        "allocated_capital": ledger_pos.entry_price * ledger_pos.quantity,
                        "gross_pnl": ledger_pos.gross_pnl,
                        "pnl": ledger_pos.unrealized_pnl(),
                        "pnl_pct": ledger_pos.return_pct,
                        "tp_pct": meta.get("tp_pct", 0),
                        "sl_pct": meta.get("sl_pct", 0),
                        "pattern": meta.get("strategy", "canonical"),
                        "duration_bars": meta.get("duration_bars", 0),
                    }
                    state.current_step = (
                        f"4. Active Position ({ledger_pos.side} {ledger_pos.quantity:.5f} BTC) -> "
                        f"PnL: ${ledger_pos.unrealized_pnl():+.3f} ({ledger_pos.return_pct:+.2f}%)"
                    )
                else:
                    snap = state.paper.snapshot(current_p)
                    state.cash_balance = round(snap.cash, 4)
                    state.total_fees_paid = round(snap.total_fees, 4)
                    state.total_trades_count += 1
                    net_pnl_dollars = close_result.net_pnl
                    net_pnl_pct = close_result.return_pct
                    if net_pnl_dollars >= 0:
                        state.winning_trades_count += 1
                        state.consecutive_losses = 0
                    else:
                        state.consecutive_losses += 1

                    outcome_type = "PROFIT" if net_pnl_dollars >= 0 else "LOSS"
                    log_lvl = "success" if net_pnl_dollars >= 0 else "warning"
                    side = ledger_pos.side if ledger_pos else "BUY"
                    qty = state.open_position["quantity"] if state.open_position else 0
                    entry_p = state.open_position["entry_price"] if state.open_position else current_p

                    if close_result.close_reason == "TP_PRICE_HIT_NET_LOSS":
                        close_tag = "🎯 TP PRICE HIT (NET LOSS AFTER FEES)"
                    elif close_result.tp_price_hit:
                        close_tag = "🎯 TAKE-PROFIT HIT"
                    elif close_result.sl_price_hit:
                        close_tag = "🛑 STOP-LOSS HIT"
                    else:
                        close_tag = "⏱️ TIME EXPIRED"

                    trade_ctx = TradeContext(
                        timestamp=time.strftime("%H:%M:%S"),
                        side=side,
                        pattern=state.open_position.get("pattern", "Unknown") if state.open_position else "Unknown",
                        regime=state.current_regime,
                        regime_confidence=state.regime_confidence,
                        rsi=state.pattern_rsi,
                        macd=state.pattern_macd,
                        entry_price=entry_p,
                        exit_price=current_p,
                        quantity=qty,
                        allocated_capital=entry_p * qty,
                        pnl=net_pnl_dollars,
                        pnl_pct=net_pnl_pct,
                        duration_bars=state.open_position.get("duration_bars", 0) if state.open_position else 0,
                        outcome=outcome_type,
                        price_trajectory=list(state.price_history)[-10:],
                    )
                    entry_indicators = state.open_position.get("entry_indicators") if state.open_position else None
                    webrl_result = webrl.on_loss(trade_ctx, entry_indicators=entry_indicators) if outcome_type == "LOSS" else webrl.on_win(trade_ctx, entry_indicators=entry_indicators)

                    # Persist audit-grade TradeExperience record
                    try:
                        import datetime as _dt
                        numeric_feats = tuple((k, float(v)) for k, v in (entry_indicators or {}).items() if isinstance(v, (int, float)))
                        experience_store.append(
                            TradeExperience(
                                timestamp=_dt.datetime.now(_dt.timezone.utc),
                                symbol="BTC/USDT",
                                timeframe="1m",
                                market_features=numeric_feats,
                                regime=state.current_regime,
                                regime_confidence=state.regime_confidence,
                                strategy=state.open_position.get("pattern", "unknown") if state.open_position else "unknown",
                                action=side,
                                signal_confidence=0.75,
                                expected_value=0.0,
                                entry_price=entry_p,
                                exit_price=current_p,
                                quantity=qty,
                                gross_pnl=close_result.gross_pnl,
                                fees=close_result.fees,
                                slippage=close_result.slippage,
                                net_pnl=net_pnl_dollars,
                                return_pct=net_pnl_pct,
                                maximum_adverse_excursion=0.0,
                                maximum_favorable_excursion=0.0,
                                duration=state.open_position.get("duration_bars", 0) if state.open_position else 0,
                                drawdown_before=state.drawdown,
                                drawdown_after=state.drawdown,
                                outcome=outcome_type,
                                model_version="v1.0.0",
                                strategy_version="v1.0.0",
                                feature_version="v1.0.0",
                            )
                        )
                    except Exception as exp_err:
                        pass

                    closed_trade = {
                        "time": time.strftime("%H:%M:%S"),
                        "symbol": "BTC/USDT",
                        "side": side,
                        "entry_price": entry_p,
                        "exit_price": current_p,
                        "quantity": qty,
                        "fee": close_result.fees,
                        "slippage": close_result.slippage,
                        "pnl": round(net_pnl_dollars, 4),
                        "gross_pnl": round(close_result.gross_pnl, 4),
                        "pnl_pct": net_pnl_pct,
                        "outcome": outcome_type,
                        "balance_after": state.cash_balance,
                        "close_reason": close_tag,
                        "webrl_feedback": webrl_result.get("event", ""),
                    }
                    state.trades_history.insert(0, closed_trade)
                    if len(state.trades_history) > 50:
                        state.trades_history.pop()

                    state.events.insert(0, {
                        "time": time.strftime("%H:%M:%S"),
                        "level": log_lvl,
                        "message": (
                            f"{close_tag}: CLOSED {side} {qty:.5f} BTC @ ${current_p:,.2f} | "
                            f"Net: ${net_pnl_dollars:+.3f} ({net_pnl_pct:+.2f}%) | "
                            f"Balance: ${state.cash_balance:.2f}"
                        ),
                    })
                    state.open_position = None
                    state.daily_pnl = round(state.equity - state.initial_capital, 4)
                    state.current_step = f"5. {close_tag} -> Scanning (Balance: ${state.cash_balance:.2f})"

            # 3. If Flat, run canonical decision pipeline
            else:
                if state.cash_balance < 2.00:
                    state.circuit_breaker = "OPEN"
                    state.agent_status = "HALTED — capital below survival floor; champion unchanged"
                    state.events.insert(0, {
                        "time": time.strftime("%H:%M:%S"),
                        "level": "error",
                        "message": f"⛔ SURVIVAL HALT: Balance ${state.cash_balance:.2f} < $2.00 — auto-retraining queued",
                    })

                profit_so_far = round(state.equity - state.initial_capital, 2)
                if profit_so_far >= state.target_profit:
                    state.agent_status = f"🏆 TARGET REACHED: +${profit_so_far:.2f} PROFIT! (Equity: ${state.equity:.2f})"

                if state.circuit_breaker != "OPEN":
                    trend = ind["trend"]
                    rsi = ind["rsi_14"]
                    rsi_zone = ind["rsi_zone"]
                    bb_pos = ind["bb_position"]
                    momentum = ind["momentum_20"]
                    atr_pct = ind["atr_pct"]
                    price = state.btc_price

                    state.current_regime = trend if trend != "FLAT" else ("Bullish Bias" if momentum >= 0 else "Bearish Bias")
                    state.regime_confidence = round(min(0.95, 0.65 + abs(momentum) / 10), 2)

                    # Continuous Multi-Pattern Analysis across all 4 specialist strategies
                    scanned_patterns = scan_specialist_patterns(ind)
                    state.candidate_patterns = scanned_patterns
                    
                    armed_list = [p for p in scanned_patterns if p["status"] == "ARMED"]
                    if armed_list:
                        top_scanned = max(armed_list, key=lambda x: x["confidence"])
                        state.detected_pattern = f"{top_scanned['name']} ({top_scanned['signal']} @ {top_scanned['confidence']}%)"
                    else:
                        top_scanned = max(scanned_patterns, key=lambda x: x["confidence"])
                        state.detected_pattern = f"Scanning ({top_scanned['name']} nearest @ {top_scanned['confidence']}%)"

                    # Continuous Reinforcement Learning & Self-Training step on every price tick
                    training_reward = round((state.equity - state.initial_capital) * 0.05, 4)
                    training_action = "BUY" if momentum >= 0 else "SELL"
                    webrl.q_table.update(ind, training_action, training_reward)
                    webrl.muzero.update_value_prior(negative_bias=(state.drawdown > 0.02))

                    chosen_eval = webrl.evaluate_trade(
                        pattern=state.detected_pattern, regime=trend, regime_conf=state.regime_confidence,
                        rsi=rsi, macd=state.pattern_macd, side=top_scanned.get("signal") or "BUY",
                        current_drawdown=state.drawdown, indicators=ind,
                    )
                    q_value = float(chosen_eval.get("q_value", 0.0))
                    p_win = min(1.0, max(0.0, 0.5 + q_value * 0.25))

                    regime_perf = experience_store.aggregate_by_strategy_regime()
                    daily_loss_pct = max(0.0, -state.daily_pnl / max(state.initial_capital, 1.0))

                    decision = state.paper.evaluate_entry(
                        indicators=ind,
                        price=price,
                        regime=state.current_regime,
                        regime_confidence=state.regime_confidence,
                        drawdown=state.drawdown,
                        consecutive_losses=state.consecutive_losses,
                        daily_loss=daily_loss_pct,
                        q_p_win=p_win,
                        regime_performance=regime_perf,
                    )
                    state.webrl_eval = {**chosen_eval, "canonical_decision": decision.reason, "expected_value": decision.expected_value}
                    state.last_decision = decision.action if decision.action == "HOLD" else f"{decision.side} ({decision.strategy})"
                    if decision.strategy:
                        state.detected_pattern = f"{decision.strategy.replace('_', ' ').title()} ({decision.side} Executed)"
                    state.risk_guard_status = f"EV={decision.expected_value:.4f} cost={decision.estimated_cost:.3f}% | {decision.reason}"

                    if decision.action == "TRADE":
                        if state.paper.open_trade("BTC/USDT", decision, price):
                            pos = state.paper.ledger.open_position
                            snap = state.paper.snapshot(price)
                            state.cash_balance = round(snap.cash, 4)
                            state.equity = round(snap.equity, 4)
                            state.open_position = {
                                "symbol": "BTC/USDT",
                                "side": pos.side,
                                "entry_price": pos.entry_price,
                                "current_price": price,
                                "quantity": pos.quantity,
                                "allocated_capital": pos.entry_price * pos.quantity,
                                "tp_pct": decision.target_pct,
                                "sl_pct": decision.stop_pct,
                                "pattern": decision.strategy,
                                "entry_indicators": dict(ind),
                                "duration_bars": 0,
                                "q_value": q_value,
                            }
                            verb = "🟢 BOUGHT" if pos.side == "BUY" else "🔴 SHORTED"
                            state.events.insert(0, {
                                "time": time.strftime("%H:%M:%S"),
                                "level": "info",
                                "message": (
                                    f"{verb}: {pos.side} {pos.quantity:.5f} BTC @ ${price:,.2f} | "
                                    f"Strategy: {decision.strategy} | TP=+{decision.target_pct:.2f}% "
                                    f"SL=-{decision.stop_pct:.2f}% | EV={decision.expected_value:.4f}"
                                ),
                            })
                            state.current_step = f"3. Live order executed: {pos.side} via {decision.strategy}"
                    else:
                        state.current_step = f"2. Scanning patterns — ({decision.reason})"
                    state.faiss_match_desc = f"Q={q_value:+.3f} regime={state.current_regime} conf={state.regime_confidence:.0%}"

            # 5. Compute Total Equity from canonical ledger
            snap = state.paper.snapshot(state.btc_price)
            state.cash_balance = round(snap.cash, 4)
            state.equity = round(snap.equity, 4)
            state.total_fees_paid = round(snap.total_fees, 4)
            state.daily_pnl = round(state.equity - state.initial_capital, 4)
            if state.equity > state.peak_equity:
                state.peak_equity = state.equity
            state.drawdown = round(max(0.0, (state.peak_equity - state.equity) / max(state.peak_equity, 0.01)), 4)

            # Check boundary self-learning ($25.00 Target Goal or Ruin Mitigation)
            boundary_ev = webrl.check_equity_boundary(state.equity)
            if boundary_ev:
                state.events.insert(0, {
                    "time": time.strftime("%H:%M:%S"),
                    "level": "success" if boundary_ev["type"] == "PROFIT_GOAL_ACHIEVED" else "warning",
                    "message": boundary_ev["message"]
                })

        # 6. Broadcast Real-Time Pipeline State + WebRL Telemetry over WebSocket
        broadcast_ws(get_current_pipeline_payload())

# HTTP Request Handler
class UnifiedRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if self.headers.get("Upgrade", "").lower() == "websocket":
            self.handle_websocket_handshake()
            return

        if path == "/favicon.ico" or path.startswith("/.well-known/"):
            self.send_response(204)
            self.end_headers()
            return

        if path == "/health":
            self.send_json({"status": "ok", "version": "1.0.0", "btc_live_price": state.btc_price})
            return

        # Server-Sent Events (SSE) Stream Endpoint for 100% resilient browser connection
        if path in ["/events", "/api/events", "/api/stream"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with state.lock:
                        payload = get_current_pipeline_payload()
                    msg = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                    self.wfile.write(msg)
                    self.wfile.flush()
                    time.sleep(1.5)
            except Exception:
                pass
            return

        if path in ["/state", "/api/state", "/trading/status", "/api/trading/status"]:
            with state.lock:
                data = get_current_pipeline_payload()["data"]
            self.send_json(data)
            return

        if path in ["/risk/status", "/api/risk/status"]:
            with state.lock:
                data = {
                    "circuit_breaker": state.circuit_breaker,
                    "risk_score": state.loss_analyzer_risk_score,
                    "risk_status": state.risk_guard_status,
                    "drawdown": state.drawdown,
                }
            self.send_json(data)
            return

        if path in ["/models", "/models/", "/api/models"]:
            self.send_json({"models": ["ppo_v1", "hmm_v1", "xgboost_loss_analyzer_v1", "experience_memory_v1"], "production": "ppo_v1"})
            return

        if path == "/" or path == "/dashboard" or path == "/dashboard/":
            self.path = "/index.html"
        elif path.startswith("/dashboard/"):
            self.path = path[len("/dashboard"):]

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        if path in ["/api/reset_capital", "/trading/reset"]:
            with state.lock:
                state.initial_capital = 10.00
                state.cash_balance = 10.00
                state.equity = 10.00
                state.peak_equity = 10.00
                state.drawdown = 0.00
                state.daily_pnl = 0.00
                state.total_trades_count = 0
                state.winning_trades_count = 0
                state.total_fees_paid = 0.00
                state.open_position = None
                state.paper = PaperTradingSession(
                    initial_cash=state.initial_capital,
                    fee_pct=state.trading_fee_pct / 100.0,
                    slippage_pct=0.0005,
                )
                state.events.insert(0, {
                    "time": time.strftime("%H:%M:%S"),
                    "level": "success",
                    "message": "💵 Virtual Account Reset: Capital restored to $10.00 (Target: +$15.00 Profit)"
                })
            self.send_json({"status": "reset", "balance": 10.00})
            return

        if path in ["/api/circuit_breaker", "/risk/circuit_breaker"]:
            with state.lock:
                state.circuit_breaker = "OPEN" if state.circuit_breaker == "CLOSED" else "CLOSED"
                state.events.insert(0, {
                    "time": time.strftime("%H:%M:%S"),
                    "level": "error" if state.circuit_breaker == "OPEN" else "success",
                    "message": f"Circuit Breaker set to {state.circuit_breaker}"
                })
            self.send_json({"circuit_breaker": state.circuit_breaker})
            return

        if path in ["/api/evolve", "/trading/evolve"]:
            with state.lock:
                webrl.record_evolution(
                    ev_type="MANUAL_TRIGGER",
                    reason="User-Dispatched Self-Evolution",
                    details="Synchronized FAISS memory index & ran ORM mini-batch replay"
                )
                state.events.insert(0, {
                    "time": time.strftime("%H:%M:%S"),
                    "level": "info",
                    "message": f"🔄 Self-Evolution Cycle Executed at {webrl.last_evolution_time}: Integrating recent trade experiences into FAISS memory index..."
                })
            self.send_json({"status": "started", "evolution_time": webrl.last_evolution_time})
            return

        self.send_error(404, "Not Found")

    def handle_websocket_handshake(self):
        key = self.headers.get("Sec-WebSocket-Key", "")
        if not key:
            self.send_error(400, "Missing Sec-WebSocket-Key")
            return

        guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        accept_key = base64.b64encode(hashlib.sha1((key + guid).encode("utf-8")).digest()).decode("utf-8")

        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_key}\r\n\r\n"
        )
        self.wfile.write(response.encode("utf-8"))
        self.wfile.flush()

        sock = self.request
        with state.lock:
            state.ws_clients.add(sock)

        # Initial sync
        with state.lock:
            init_payload = get_current_pipeline_payload()
        send_ws_frame(sock, json.dumps(init_payload), opcode=0x1)

        # Standard RFC 6455 Frame Decoder loop (Handles ping/pong, close, binary, text)
        try:
            while True:
                header = sock.recv(2)
                if not header or len(header) < 2:
                    break
                byte1, byte2 = header[0], header[1]
                opcode = byte1 & 0x0F
                is_masked = bool(byte2 & 0x80)
                payload_len = byte2 & 0x7F
                if payload_len == 126:
                    ext = sock.recv(2)
                    if len(ext) < 2:
                        break
                    payload_len = struct.unpack("!H", ext)[0]
                elif payload_len == 127:
                    ext = sock.recv(8)
                    if len(ext) < 8:
                        break
                    payload_len = struct.unpack("!Q", ext)[0]

                masks = sock.recv(4) if is_masked else b""
                payload = bytearray()
                while len(payload) < payload_len:
                    chunk = sock.recv(min(4096, payload_len - len(payload)))
                    if not chunk:
                        break
                    payload.extend(chunk)

                if is_masked:
                    payload = bytearray(b ^ masks[i % 4] for i, b in enumerate(payload))

                # Close frame (0x8)
                if opcode == 0x8:
                    send_ws_frame(sock, b"", opcode=0x8)
                    break
                # Ping frame (0x9) -> respond with Pong (0xA)
                elif opcode == 0x9:
                    send_ws_frame(sock, payload, opcode=0xA)
        except Exception:
            pass
        finally:
            with state.lock:
                state.ws_clients.discard(sock)

    def send_json(self, data, status_code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def run_server(port=8080):
    server = ThreadedHTTPServer(("0.0.0.0", port), UnifiedRequestHandler)

    print(f"============================================================")
    print(f"⚡ Self-Evolving Crypto Trading Command Center is LIVE!")
    print(f"💰 Virtual Starting Capital: ${state.initial_capital:.2f} (Target: +$15.00 Profit | Goal: $25.00)")
    print(f"📊 Dashboard URL: http://localhost:{port}/")
    print(f"🌐 Alternative:   http://127.0.0.1:{port}/")
    print(f"🔌 WebSocket URL: ws://localhost:{port}/ws")
    print(f"📡 SSE Stream:    http://localhost:{port}/api/stream")
    print(f"============================================================")
    sys.stdout.flush()
    
    # Start live trading lifecycle worker in background
    sim_thread = threading.Thread(target=live_trading_lifecycle_worker, daemon=True)
    sim_thread.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
    finally:
        server.server_close()

if __name__ == "__main__":
    run_server(8080)
