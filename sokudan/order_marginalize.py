"""Order marginalisation at inference time (docs/order_marginalization.md).

A question is asked again with its options in other orders, and each answer is mapped
back to the original order before the probabilities are averaged. No training; the
encoding of a question depends only on its instructions and the option texts in order
(`sokudan.encoding.question.encode_joint`), so a reordered question is the same
question with its option texts reordered.

- `choice` (K options): the K cyclic shifts.
- `score` (K levels): the original order and its reverse. An ordinal scale is not
  rotated; the ordinal head outputs one probability per level, so the reversed
  distribution is mapped back level by level.
- `bool`: the two slots in the original order (false, true) and swapped (true, false).
  The swapped question is rendered as a two-option choice with the true label first --
  the same tokens a bool with its slots swapped would produce -- and scored unordered,
  as a bool is.

Which types are marginalised can be chosen per type (`marginalized_types`,
docs/order_marginalization.md §7); the variants of each type are the same either way.

`variants(question)` gives `(question to encode, perm)` pairs, where `perm[slot]` is the
original index of the option shown in that slot; `combine(rows, perms)` maps each row
back (`p[perm[slot]] = row[slot]`) and takes the arithmetic mean. The first variant is
always the question itself with the identity permutation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from sokudan.schema.question import BoolQuestion, ChoiceQuestion, Question, ScoreQuestion

TYPES = ("choice", "score", "bool")


def marginalized_types(flag: bool | Mapping[str, bool] | None) -> frozenset[str]:
    """The question types to marginalise: `True` is every type, `False` / `None` none, and
    a mapping such as `{"choice": False, "score": True, "bool": False}` picks per type
    (types left out are off). Unknown types or non-bool values are refused."""
    if flag is None or flag is False:
        return frozenset()
    if flag is True:
        return frozenset(TYPES)
    if not isinstance(flag, Mapping):
        raise TypeError("order_marginalize takes a bool or a {type: bool} mapping")
    unknown = set(flag) - set(TYPES)
    if unknown:
        raise ValueError(f"unknown question types {sorted(unknown)}; expected {TYPES}")
    if not all(isinstance(v, bool) for v in flag.values()):
        raise TypeError("order_marginalize values must be bool")
    return frozenset(t for t, on in flag.items() if on)


def reordered(question: Question, perm: Sequence[int]) -> Question:
    """`question` with its options shown in the order `perm` (`perm[slot]` = original
    index). The one place a question is reordered: order marginalisation (`variants`) and
    order augmentation in training (`sokudan.train.order_augment`) both come here."""
    if isinstance(question, ChoiceQuestion):
        items = list(question.criteria.items())
        pairs = [items[i] for i in perm]
        # A dict for real questions; a probe may carry repeated labels in a list-like
        # `criteria` (scripts/presentation_checks_sokudan.py), kept as its own type.
        criteria = (dict(pairs) if isinstance(question.criteria, dict)
                    else type(question.criteria)(pairs))
        return question.model_copy(update={"criteria": criteria})
    if isinstance(question, ScoreQuestion):
        return question.model_copy(update={"criteria": [question.criteria[i] for i in perm]})
    if isinstance(question, BoolQuestion):
        if list(perm) == [0, 1]:
            return question
        if list(perm) != [1, 0]:
            raise ValueError("a bool has two slots")
        return ChoiceQuestion.model_construct(
            type="choice", instructions=question.instructions,
            criteria={question.true_label: "", question.false_label: ""})
    raise TypeError(f"no reordering for {type(question).__name__}")


def variants(question: Question) -> list[tuple[Question, list[int]]]:
    if isinstance(question, ChoiceQuestion):
        k = len(question.criteria)
        out: list[tuple[Question, list[int]]] = [(question, list(range(k)))]
        for shift in range(1, k):
            perm = [(slot + shift) % k for slot in range(k)]
            out.append((reordered(question, perm), perm))
        return out
    if isinstance(question, ScoreQuestion):
        k = len(question.criteria)
        perm = list(reversed(range(k)))
        return [(question, list(range(k))), (reordered(question, perm), perm)]
    if isinstance(question, BoolQuestion):
        return [(question, [0, 1]), (reordered(question, [1, 0]), [1, 0])]
    raise TypeError(f"no order variants for {type(question).__name__}")


def combine(rows: Sequence[np.ndarray], perms: Sequence[Sequence[int]]) -> np.ndarray:
    """Map every row back to the original option order and average them."""
    if len(rows) != len(perms) or not rows:
        raise ValueError("one permutation per row, at least one row")
    k = len(perms[0])
    total = np.zeros(k, dtype=np.float64)
    for row, perm in zip(rows, perms, strict=True):
        row = np.asarray(row, dtype=np.float64)
        if len(row) != k or sorted(perm) != list(range(k)):
            raise ValueError("every row and permutation must cover the same K options")
        back = np.empty(k, dtype=np.float64)
        back[list(perm)] = row
        total += back
    return total / len(rows)
