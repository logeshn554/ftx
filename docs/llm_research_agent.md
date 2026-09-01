# LLM Research Agent

Role: **research scientist only**.

May receive aggregated statistical reports and output structured hypotheses:
`hypothesis`, `rationale`, `proposed_change`, `expected_effect`, `risk`, `test_plan`.

Must NEVER:
- Submit orders
- Modify production weights
- Bypass risk or circuit breakers
- Promote models
- Declare profitability without validation

Implementation: `src/trade/evolution/llm_researcher.py` with deterministic fallback when LLM unavailable.
