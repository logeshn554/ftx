from .base import Signal, abstain
class TrendStrategy:
    name = "trend"
    def signal(self, i: dict) -> Signal:
        fast, slow, adx = float(i.get("sma_10", 0)), float(i.get("sma_50", 0)), float(i.get("adx", 0))
        if not fast or not slow or adx < 25: return abstain(self.name)
        side = "BUY" if fast > slow else "SELL"
        move = abs(fast - slow) / max(slow, 1e-12)
        return Signal(side, min(1., .5 + adx / 100), move, move / 2, move, self.name)
