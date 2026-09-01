from .base import Signal, abstain
class MeanReversionStrategy:
    name = "mean_reversion"
    def signal(self, i: dict) -> Signal:
        rsi = float(i.get("rsi_14", 50)); pct = float(i.get("bb_pct", .5))
        if rsi < 30 and pct < .2: side, confidence = "BUY", min(1., (30-rsi)/20+.5)
        elif rsi > 70 and pct > .8: side, confidence = "SELL", min(1., (rsi-70)/20+.5)
        else: return abstain(self.name)
        move = abs(rsi - 50) / 100
        return Signal(side, confidence, move, move / 2, move, self.name)
