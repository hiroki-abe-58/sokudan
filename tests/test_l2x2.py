"""The decision rules of docs/length_2x2.md §6, on made-up numbers.

The rules are applied by `scripts/l2x2_gate.py` after the runs; these pin down that
the code says what the pre-registration says before any real number goes through it.
"""

from __future__ import annotations

from scripts.l2x2_build import JACCARD, grams
from scripts.l2x2_gate import candidate, differs, result_type, summarise

BASE = {"M1": 0.89, "M5": 0.80, "choice_accuracy": 0.705, "score_rps": 0.178,
        "score_accuracy": 0.524, "bool_accuracy": 0.931}


def cond(m2: list[float], **over: float) -> dict:
    seeds = [{**BASE, **over, "M2": v} for v in m2]
    return summarise(seeds)


def base_cond(m2: list[float]) -> dict:
    # A v0.1-like condition whose own values set the cut (min / max of these seeds).
    seeds = []
    for i, v in enumerate(m2):
        seeds.append({"M1": 0.88 + 0.005 * i, "M5": 0.77 + 0.02 * i,
                      "choice_accuracy": 0.70 + 0.005 * i, "score_rps": 0.19 - 0.01 * i,
                      "score_accuracy": 0.51 + 0.01 * i, "bool_accuracy": 0.929 + 0.001 * i,
                      "M2": v})
    return summarise(seeds)


def test_difference_needs_to_exceed_the_larger_range():
    a, b = cond([0.80, 0.82, 0.84]), cond([0.85, 0.855, 0.865])
    sig, diff = differs(b, a)  # 0.8567 - 0.82 = 0.0367, ranges 0.04 / 0.015
    assert not sig and diff > 0
    c = cond([0.87, 0.88, 0.89])  # 0.06 > 0.04
    assert differs(c, a)[0]


def test_type_a_length():
    s = {"qf-short": cond([0.83, 0.845, 0.855]), "sf-short": cond([0.875, 0.885, 0.89]),
         "qf-long": cond([0.89, 0.895, 0.90]), "sf-long": cond([0.89, 0.895, 0.90])}
    kind, holds = result_type(s)
    assert kind == "A"
    assert "C" not in holds


def test_type_b_position():
    s = {"qf-short": cond([0.83, 0.845, 0.855]), "sf-short": cond([0.875, 0.885, 0.89]),
         "qf-long": cond([0.835, 0.85, 0.86]), "sf-long": cond([0.88, 0.885, 0.89])}
    assert result_type(s)[0] == "B"


def test_type_c_both():
    s = {"qf-short": cond([0.80, 0.805, 0.81]), "sf-short": cond([0.84, 0.845, 0.85]),
         "qf-long": cond([0.86, 0.865, 0.87]), "sf-long": cond([0.92, 0.925, 0.93])}
    kind, holds = result_type(s)
    # qf-long > sf-short is 差あり, so A holds too and comes first; C is recorded.
    assert kind == "A" and "C" in holds


def test_type_d_otherwise():
    s = {"qf-short": cond([0.83, 0.845, 0.855]), "sf-short": cond([0.84, 0.85, 0.86]),
         "qf-long": cond([0.84, 0.85, 0.86]), "sf-long": cond([0.84, 0.85, 0.86])}
    assert result_type(s)[0] == "D"


def test_candidate_prefers_the_simpler_of_tied_conditions():
    s = {"qf-short": base_cond([0.83, 0.845, 0.855]),
         "sf-short": cond([0.875, 0.885, 0.89]),
         "qf-long": cond([0.88, 0.886, 0.89]),
         "sf-long": cond([0.87, 0.884, 0.89])}
    chosen, detail = candidate(s)
    # best M2 is qf-long; sf-short and sf-long are within range of it; qf-long is the
    # simplest of the tied set (question_first before state_first).
    assert detail["best_M2"] == "qf-long"
    assert chosen == "qf-long"


def test_candidate_falls_back_to_v01_when_only_it_passes():
    s = {"qf-short": base_cond([0.83, 0.845, 0.855]),
         "sf-short": cond([0.875, 0.885, 0.89], M5=0.70)}  # M5 below v0.1's min 0.77
    chosen, detail = candidate(s)
    assert detail["passing"] == ["qf-short"]
    assert chosen.startswith("qf-short")


def test_rps_cut_is_upper_bound():
    s = {"qf-short": base_cond([0.83, 0.845, 0.855]),
         "sf-short": cond([0.90, 0.905, 0.91], score_rps=0.20)}  # worse than v0.1 max 0.19
    assert not candidate(s)[1]["cuts"]["sf-short"]["score_rps"]


def test_char_5gram_jaccard():
    text = "お問い合わせの件について、担当者から折り返しご連絡いたします。"
    a, b = grams(text), grams(text + "よろしくお願いいたします。")
    jac = len(a & b) / len(a | b)
    assert jac >= JACCARD
    c = grams("まったく別の文面で、共通する語句はほとんどありません。")
    assert len(a & c) / len(a | c) < JACCARD
