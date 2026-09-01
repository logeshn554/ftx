"""Optional LLM research agent — hypothesis generation only, never trading authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResearchHypothesis:
    hypothesis: str
    rationale: str
    proposed_change: dict[str, Any]
    expected_effect: str
    risk: str
    test_plan: str


class LLMResearcher:
    """Deterministic fallback when no LLM is available."""

    def __init__(self, llm_available: bool = False):
        self.llm_available = llm_available

    def generate_hypothesis(self, report: dict[str, Any]) -> ResearchHypothesis:
        strategy = report.get("strategy", "unknown")
        regime = report.get("regime", "unknown")
        expectancy = float(report.get("expectancy", 0.0))
        sample_count = int(report.get("trades", report.get("sample_count", 0)))

        if sample_count < 30:
            return ResearchHypothesis(
                hypothesis="insufficient_evidence",
                rationale=f"Only {sample_count} trades; cannot infer causal pattern.",
                proposed_change={},
                expected_effect="none",
                risk="low",
                test_plan="Collect more independent observations before changing parameters.",
            )

        if expectancy < 0:
            key = f"disable_{strategy}_in_{regime}"
            return ResearchHypothesis(
                hypothesis=key,
                rationale=f"{strategy} in {regime} shows negative expectancy ({expectancy:.4%}) over {sample_count} trades.",
                proposed_change={f"strategies.{strategy}.enabled_in_{regime}": False},
                expected_effect="Reduce negative-EV exposure in weak regime.",
                risk="May miss recovery if regime shifts; validate walk-forward.",
                test_plan="Generate candidate, walk-forward OOS, cost stress, compare to champion.",
            )

        return ResearchHypothesis(
            hypothesis="maintain_current_configuration",
            rationale="No statistically meaningful degradation detected.",
            proposed_change={},
            expected_effect="preserve champion",
            risk="low",
            test_plan="Continue monitoring.",
        )

    def from_llm(self, _prompt: str) -> ResearchHypothesis | None:
        if not self.llm_available:
            return None
        return None
