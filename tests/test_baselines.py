"""Tests for the bench_ja baselines.

The random baseline is tested explicitly because the obvious implementation is
wrong: `np.full((n, k), 1/k)` has every row tied, `argmax` resolves ties to index 0,
and the "random" baseline then scores the first option's base rate -- silently
identical to the majority-class baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from sokudan.eval.baselines import (
    PROB_FLOOR,
    MajorityClassBaseline,
    RandomBaseline,
    _parse_one_word,
    floor_and_renormalise,
)
from sokudan.eval.bench_ja import DEPARTMENTS, URGENCY_LEVELS, BenchItem, bench_questions


def make_items(labels: list[tuple[str, int, bool]]) -> list[BenchItem]:
    return [
        BenchItem(
            item_id=f"t{i}", state="ダミー本文", department=d, urgency=u, churn=c,
            industry="", role="", style="", length="", channel="",
            generator_model="test", generator_temperature=0.0,
        )
        for i, (d, u, c) in enumerate(labels)
    ]


# --------------------------------------------------------------------------
# probability floor
# --------------------------------------------------------------------------

def test_floor_removes_exact_zeros_and_keeps_rows_normalised() -> None:
    out = floor_and_renormalise(np.array([[1.0, 0.0, 0.0, 0.0]]))
    assert out.min() > 0.0
    assert out.min() == pytest.approx(PROB_FLOOR, rel=1e-3)
    assert out.sum(axis=1)[0] == pytest.approx(1.0)


def test_floor_barely_moves_an_already_valid_row() -> None:
    row = np.array([[0.5, 0.3, 0.2]])
    np.testing.assert_allclose(floor_and_renormalise(row), row, atol=1e-6)


# --------------------------------------------------------------------------
# random baseline
# --------------------------------------------------------------------------

def test_random_baseline_argmax_is_not_pinned_to_the_first_option() -> None:
    items = make_items([("請求", 0, False)] * 2000)
    out = RandomBaseline(seed=0).run(items)
    share = np.bincount(out.choice_probs.argmax(axis=1), minlength=len(DEPARTMENTS)) / 2000
    assert share.max() < 0.35, f"argmax is skewed: {share}"
    assert share.min() > 0.15, f"argmax is skewed: {share}"


def test_random_baseline_confidence_stays_at_one_over_k() -> None:
    items = make_items([("請求", 0, False)] * 200)
    out = RandomBaseline(seed=0).run(items)
    assert out.choice_probs.max(axis=1).mean() == pytest.approx(1 / len(DEPARTMENTS), abs=1e-4)
    assert out.score_probs.max(axis=1).mean() == pytest.approx(1 / len(URGENCY_LEVELS), abs=1e-4)


def test_random_baseline_is_reproducible() -> None:
    items = make_items([("請求", 0, False)] * 50)
    a = RandomBaseline(seed=7).run(items)
    b = RandomBaseline(seed=7).run(items)
    np.testing.assert_array_equal(a.choice_probs, b.choice_probs)


# --------------------------------------------------------------------------
# majority-class baseline
# --------------------------------------------------------------------------

def test_majority_baseline_predicts_the_most_common_class() -> None:
    items = make_items(
        [("請求", 0, False)] * 6 + [("技術", 1, True)] * 3 + [("営業", 2, False)] * 1
    )
    out = MajorityClassBaseline().run(items)
    assert out.choice_probs.argmax(axis=1)[0] == list(DEPARTMENTS).index("請求")


def test_majority_baseline_probabilities_are_the_empirical_prior() -> None:
    items = make_items([("請求", 0, False)] * 3 + [("技術", 0, False)] * 1)
    out = MajorityClassBaseline().run(items)
    dept_index = {k: i for i, k in enumerate(DEPARTMENTS)}
    assert out.choice_probs[0, dept_index["請求"]] == pytest.approx(0.75)
    assert out.choice_probs[0, dept_index["技術"]] == pytest.approx(0.25)
    assert out.choice_probs[0, dept_index["営業"]] == pytest.approx(0.0)


def test_majority_baseline_rows_are_identical_for_every_item() -> None:
    items = make_items([("請求", 0, False), ("技術", 1, True), ("営業", 2, False)])
    out = MajorityClassBaseline().run(items)
    assert np.allclose(out.choice_probs, out.choice_probs[0])


# --------------------------------------------------------------------------
# one-word parsing for the LLM baseline
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("請求", 0),
        ("  技術。 ", 1),
        ("**営業**", 2),
        ("「その他」", 3),
        ("この件は営業が担当すべきです", 2),
        ("わかりません", None),
        ("請求か技術", None),
        ("", None),
    ],
)
def test_parse_one_word(reply: str, expected: int | None) -> None:
    assert _parse_one_word(reply, list(DEPARTMENTS)) == expected


# --------------------------------------------------------------------------
# the question schema itself
# --------------------------------------------------------------------------

def test_bench_questions_cover_all_three_primitives() -> None:
    q = bench_questions()
    assert {q[k]["type"] for k in q} == {"choice", "score", "noul"}


def test_bench_questions_options_match_the_label_spaces() -> None:
    q = bench_questions()
    assert list(q["department"]["criteria"]) == list(DEPARTMENTS)
    assert q["urgency"]["criteria"] == URGENCY_LEVELS


def test_bench_questions_are_a_fresh_object_each_call() -> None:
    """Callers mutate question dicts; a shared default would leak between runs."""
    a, b = bench_questions(), bench_questions()
    a["department"]["criteria"]["dummy"] = "x"
    assert "dummy" not in b["department"]["criteria"]
