"""Order marginalisation at inference (docs/order_marginalization.md).

- With `order_marginalize=False` (and by default) `predict` returns exactly what it did
  before the flag existed (`tests/fixtures/predict_golden.json`, recorded with that code).
- `variants` / `combine`: every reordering is mapped back to the original option order.
  A scorer that depends only on option content is left unchanged; one that depends only on
  position becomes uniform for `choice` (all cyclic shifts) and symmetric for `score`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sokudan.order_marginalize import combine, variants
from sokudan.schema.question import BoolQuestion, ChoiceQuestion, ScoreQuestion, marker_texts

GOLDEN = Path(__file__).parent / "fixtures" / "predict_golden.json"


def core(out: dict) -> dict:
    """The response without v0.2.1's calibration keys (the golden outputs predate them);
    an agent built without temperatures reports `calibrated: False` and no answers."""
    assert out["calibrated"] is False and out["calibrated_answers"] == []
    return {k: v for k, v in out.items() if k not in ("calibrated", "calibrated_answers")}


def choice_q(k: int = 4) -> ChoiceQuestion:
    return ChoiceQuestion(instructions="どれか", criteria={f"L{i}": f"d{i}" for i in range(k)})


def test_choice_variants_are_the_k_cyclic_shifts():
    q = choice_q(4)
    out = variants(q)
    assert len(out) == 4 and out[0][0] is q and out[0][1] == [0, 1, 2, 3]
    texts = marker_texts(q)
    for question, perm in out:
        assert marker_texts(question) == [texts[i] for i in perm]
        assert question.instructions == q.instructions
    assert sorted(tuple(p) for _, p in out) == sorted(
        tuple((s + i) % 4 for s in range(4)) for i in range(4))


def test_score_variants_are_original_and_reversed():
    q = ScoreQuestion(instructions="緊急度", criteria=["低", "中", "高"])
    (q0, p0), (q1, p1) = variants(q)
    assert q0 is q and p0 == [0, 1, 2]
    assert marker_texts(q1) == ["高", "中", "低"] and p1 == [2, 1, 0]
    assert q1.type == "score"


def test_bool_variants_swap_the_two_slots_and_stay_unordered():
    from sokudan.schema.question import is_ordered

    q = BoolQuestion(instructions="解約を示唆しているか")
    (q0, p0), (q1, p1) = variants(q)
    assert marker_texts(q0) == ["いいえ", "はい"] and p0 == [0, 1]
    assert marker_texts(q1) == ["はい", "いいえ"] and p1 == [1, 0]
    assert is_ordered(q1) is False and q1.instructions == q.instructions


def _score(question, *, content=None, position=None) -> np.ndarray:
    """A fake model: probability from the option's text, or from its slot."""
    texts = marker_texts(question)
    if content is not None:
        w = np.array([content[t] for t in texts], dtype=float)
    else:
        w = np.array(position[:len(texts)], dtype=float)
    return w / w.sum()


def test_content_only_scorer_is_unchanged_by_marginalisation():
    q = choice_q(4)
    content = {t: w for t, w in zip(marker_texts(q), [1.0, 3.0, 2.0, 4.0], strict=True)}
    out = variants(q)
    single = _score(q, content=content)
    averaged = combine([_score(v, content=content) for v, _ in out], [p for _, p in out])
    assert np.allclose(averaged, single)


def test_position_only_scorer_becomes_uniform_for_choice():
    q = choice_q(5)
    out = variants(q)
    averaged = combine([_score(v, position=[9, 1, 1, 1, 1]) for v, _ in out],
                       [p for _, p in out])
    assert np.allclose(averaged, np.full(5, 0.2))


def test_position_only_scorer_becomes_symmetric_for_score():
    q = ScoreQuestion(instructions="緊急度", criteria=["低", "中", "高", "最高"])
    out = variants(q)
    averaged = combine([_score(v, position=[5, 1, 1, 1]) for v, _ in out], [p for _, p in out])
    assert np.allclose(averaged, averaged[::-1])


def test_combine_rejects_mismatched_rows():
    with pytest.raises(ValueError):
        combine([np.ones(3) / 3], [[0, 1]])


@pytest.mark.slow
def test_flag_off_matches_the_outputs_recorded_before_the_flag():
    from tests.fixtures.make_predict_golden import QUESTIONS, STATES, build_agent

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    agent = build_agent()
    for state, expected in zip(STATES, golden, strict=True):
        assert core(agent.predict(state, QUESTIONS)) == expected
        assert core(agent.predict(state, QUESTIONS, order_marginalize=False)) == expected
    on = agent.predict(STATES[0], QUESTIONS, order_marginalize=True)
    assert on["usage"]["backbone_passes"] == 4 + 2 + 2
    assert on["usage"]["order_marginalized"] is True
    assert abs(sum(on["answers"]["department"]["probabilities"].values()) - 1) < 1e-3
    assert abs(sum(on["answers"]["urgency"]["probabilities"].values()) - 1) < 1e-3
    assert 0.0 <= on["answers"]["churn"]["noul"] <= 1.0


# per-type flag (docs/order_marginalization.md §7) --------------------------------------

def test_marginalized_types_reads_bools_and_per_type_mappings():
    from sokudan.order_marginalize import marginalized_types

    assert marginalized_types(False) == frozenset() == marginalized_types(None)
    assert marginalized_types(True) == {"choice", "score", "bool"}
    assert marginalized_types({"choice": False, "score": True, "bool": False}) == {"score"}
    assert marginalized_types({"score": True}) == {"score"}
    assert marginalized_types({}) == frozenset()
    with pytest.raises(ValueError):
        marginalized_types({"scores": True})
    with pytest.raises(TypeError):
        marginalized_types({"score": 1})
    with pytest.raises(TypeError):
        marginalized_types("score")


@pytest.mark.slow
def test_every_type_off_matches_the_outputs_recorded_before_the_flag():
    from tests.fixtures.make_predict_golden import QUESTIONS, STATES, build_agent

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    agent = build_agent()
    for off in ({}, {"choice": False, "score": False, "bool": False}, {"score": False}):
        for state, expected in zip(STATES, golden, strict=True):
            assert core(agent.predict(state, QUESTIONS, order_marginalize=off)) == expected


@pytest.mark.slow
def test_every_type_on_is_the_same_as_true():
    from tests.fixtures.make_predict_golden import QUESTIONS, STATES, build_agent

    agent = build_agent()
    every = {"choice": True, "score": True, "bool": True}
    for state in STATES:
        assert (agent.predict(state, QUESTIONS, order_marginalize=every)
                == agent.predict(state, QUESTIONS, order_marginalize=True))


@pytest.mark.slow
def test_score_only_marginalises_the_score_question_and_asks_the_others_once():
    from tests.fixtures.make_predict_golden import QUESTIONS, STATES, build_agent

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    agent = build_agent()
    for state, expected in zip(STATES, golden, strict=True):
        out = agent.predict(state, QUESTIONS, order_marginalize={"score": True})
        assert out["usage"]["backbone_passes"] == 1 + 2 + 1
        assert out["usage"]["order_marginalized"] == ["score"]
        for qid in ("department", "churn"):  # choice and bool: asked once, as when off
            assert out["answers"][qid] == expected["answers"][qid]
        assert out["answers"]["urgency"] != expected["answers"]["urgency"]
        assert abs(sum(out["answers"]["urgency"]["probabilities"].values()) - 1) < 1e-3
