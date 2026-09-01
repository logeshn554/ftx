from trade.evolution.candidate_generator import CandidateGenerator


def test_candidate_generation_is_deterministic_and_does_not_mutate_champion():
    champion = {"training": {"reward_turnover_penalty": 0.05}, "trading": {"default_position_size_pct": 10.0}}
    original = {"training": {"reward_turnover_penalty": 0.05}, "trading": {"default_position_size_pct": 10.0}}
    a = CandidateGenerator(seed=7).generate(champion, "v1.2.3", 4)
    b = CandidateGenerator(seed=7).generate(original, "v1.2.3", 4)
    assert champion == original
    assert [(x.candidate_id, x.config) for x in a] == [(x.candidate_id, x.config) for x in b]
    assert all(x.parent_version == "v1.2.3" and x.hypothesis for x in a)


def test_candidate_mutation_stays_within_hard_bounds():
    champion = {"trading": {"default_position_size_pct": 19.99}}
    candidates = CandidateGenerator(seed=1).generate(champion, "v1", 30)
    for candidate in candidates:
        value = candidate.config.get("trading", {}).get("default_position_size_pct", 0)
        assert 0.1 <= value <= 20.0


def test_zero_candidates_is_supported():
    assert CandidateGenerator().generate({}, "v1", 0) == []
