"""Deterministic specialist strategy interfaces."""
from .base import Strategy, Signal
from .trend import TrendStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy
from .momentum import MomentumStrategy

__all__ = ["Strategy", "Signal", "TrendStrategy", "MeanReversionStrategy", "BreakoutStrategy", "MomentumStrategy"]
