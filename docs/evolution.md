# Self-Evolution

1. Champion trades (immutable)
2. Experiences stored in `ImmutableExperienceStore`
3. `EvolutionOrchestrator` triggers on evidence threshold
4. `CandidateGenerator` produces bounded one-change hypotheses
5. `CandidateEvaluator` runs walk-forward + cost stress
6. `ChampionSelector` applies multi-criteria promotion
7. Rollback restores previous champion config

Q-learning, MuZero, GRPO: research signals only — cannot override risk or promote models.

LLM (`llm_researcher.py`): optional hypothesis generator; never order authority.
