"""The selection rule of docs/release_candidate.md (scripts/release_eval.py)."""

from __future__ import annotations

from scripts.release_eval import passes, select

REF = {"choice_accuracy": 0.70, "bool_accuracy": 0.93, "score_accuracy": 0.52,
       "score_rps": 0.177, "ece_val": 0.062, "M1m": 0.83}


def entry(**changes: float) -> dict:
    return {**REF, **changes}


def test_cutoffs_sit_exactly_on_the_margins():
    edge = entry(choice_accuracy=0.695, bool_accuracy=0.925, score_accuracy=0.51,
                 score_rps=0.182, ece_val=0.082)
    assert all(passes(edge, REF).values())
    assert not passes(entry(bool_accuracy=0.9249), REF)["bool_accuracy"]
    assert not passes(entry(score_rps=0.1821), REF)["score_rps"]
    assert not passes(entry(ece_val=0.0821), REF)["ece_val"]
    assert not passes(entry(score_accuracy=0.5099), REF)["score_accuracy"]


def test_highest_m1m_among_those_that_pass():
    scores = {"S8_old": entry(M1m=0.860), "S16": entry(M1m=0.870),
              "S24": entry(M1m=0.880, bool_accuracy=0.90)}
    assert select(scores, list(scores), REF) == "S16"


def test_within_0002_the_fewest_members_win():
    scores = {"S8_old": entry(M1m=0.8650), "S16": entry(M1m=0.8665), "S24": entry(M1m=0.8670)}
    assert select(scores, list(scores), REF) == "S8_old"
    scores["S8_old"]["M1m"] = 0.8645
    assert select(scores, list(scores), REF) == "S16"


def test_equal_member_counts_fall_back_to_m1m():
    scores = {"S8_old": entry(M1m=0.8650), "S8_new": entry(M1m=0.8660)}
    assert select(scores, list(scores), REF) == "S8_new"


def test_none_pass_means_no_candidate():
    scores = {"S8_old": entry(choice_accuracy=0.60)}
    assert select(scores, ["S8_old"], REF) is None


def test_release_rule_boundaries():
    from scripts.release_bench import rule

    ref = {"choice/accuracy": 0.847, "score/rps": 0.090, "score/accuracy": 0.763,
           "bool/accuracy": 0.788, "bool/auroc": 0.789}
    edge = {"choice/accuracy": 0.837, "score/rps": 0.085, "score/accuracy": 0.753,
            "bool/accuracy": 0.778, "bool/auroc": 0.799}
    assert rule(edge, ref)["pass"]
    for key, value in (("bool/auroc", 0.7989), ("score/rps", 0.0851),
                       ("choice/accuracy", 0.8369), ("score/accuracy", 0.7529),
                       ("bool/accuracy", 0.7779)):
        assert not rule({**edge, key: value}, ref)["pass"], key
