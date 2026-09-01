from .base import Signal, abstain
class BreakoutStrategy:
    name = "breakout"
    def signal(self, i: dict) -> Signal:
        pct, width = float(i.get("bb_pct", .5)), float(i.get("bb_width", 0))
        if width <= 0 or pct < .95 and pct > .05: return abstain(self.name)
        side = "BUY" if pct >= .95 else "SELL"; move = width
        return Signal(side, min(1., .55 + width), move, move / 2, move, self.name)
