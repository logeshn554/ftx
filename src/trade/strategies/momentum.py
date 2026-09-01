from .base import Signal, abstain
class MomentumStrategy:
    name = "momentum"
    def signal(self, i: dict) -> Signal:
        roc = float(i.get("roc_10", 0)); threshold = float(i.get("momentum_threshold", .01))
        if abs(roc) < threshold: return abstain(self.name)
        move = abs(roc); return Signal("BUY" if roc > 0 else "SELL", min(1., .5 + abs(roc)), move, move / 2, move, self.name)
