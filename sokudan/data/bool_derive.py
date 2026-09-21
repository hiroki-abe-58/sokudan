"""Derive extra `bool` schemas from labels that already exist (no new generation).

Why this exists. `docs/benchmarks.md` measured `bool` AUROC at 0.888 on validation
(schema families seen in training) and 0.405 on `bench_ja` (a schema never seen) --
the ranking does not merely weaken, it inverts. Diagnostics ruled out the obvious
causes: the model does use the state (swapping documents moves accuracy by 0.26), the
pipeline memorises 100 examples to 0.970, and the P(true) polarity is consistent end
to end.

What is left is that the catalogue only contained **10** boolean schemas. Ten phrasings
is not enough variety to learn "read this yes/no question" rather than "recognise these
ten questions". The fix is to manufacture many more from gold labels already in hand:

* a `choice` attribute with K options yields K membership questions ("is this A?")
* a `score` attribute with K levels yields K-1 threshold questions ("is it at least k?")
* every boolean question has a negated twin with the label flipped

Negations are built from templates, never from an LLM: a generated negation that
subtly failed to negate would silently mislabel every example derived from it.

None of this calls the generator. It is a relabelling of documents already written.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from sokudan.data.builders.synthetic import Attribute, Domain
from sokudan.schema.question import BoolQuestion

# §1(d): the surface pair is drawn at random so the model cannot key on one spelling.
BOOL_LABEL_PAIRS: list[tuple[str, str]] = [
    ("いいえ", "はい"),
    ("該当しない", "該当する"),
    ("false", "true"),
    ("当てはまらない", "当てはまる"),
]

# Templates only. A generated negation that failed to negate would mislabel every
# example derived from it, and nothing downstream could detect that.
NEGATION_TEMPLATES: list[str] = [
    "{stem}とは言えないか",
    "{stem}に該当しないか",
    "{stem}わけではないか",
]

MEMBERSHIP_TEMPLATES: list[str] = [
    "この文書は「{option}」に該当するか",
    "{subject}は「{option}」か",
    "「{option}」と判断してよいか",
]

THRESHOLD_TEMPLATES: list[str] = [
    "{subject}は「{level}」以上か",
    "{subject}は少なくとも「{level}」に達しているか",
    "{subject}は「{level}」かそれより上か",
]


@dataclass(frozen=True)
class DerivedBool:
    """A boolean question derived from an existing labelled attribute."""

    question: BoolQuestion
    label: int              # 0 = false, 1 = true
    source_attribute: str
    kind: str               # "membership" | "threshold" | "negation" | "original"
    schema_id: str          # groups examples that share a question, for balancing


def _subject(attribute: Attribute) -> str:
    """A short noun phrase for the thing being asked about."""
    stem = attribute.instructions.rstrip("。.？?")
    for suffix in ("は", "が", "を"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _labels(rng: random.Random) -> tuple[str, str]:
    return rng.choice(BOOL_LABEL_PAIRS)


def _negate(stem: str, rng: random.Random) -> str:
    base = stem.rstrip("。.？?")
    for suffix in ("か", "は", "が"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return rng.choice(NEGATION_TEMPLATES).format(stem=base)


def derive_from_choice(
    attribute: Attribute, gold_index: int, rng: random.Random
) -> list[DerivedBool]:
    """K membership questions from one K-way choice: "is this A?" for each A.

    Exactly one comes out true, so drawing all K would hand the model a 1/K positive
    rate. The caller balances; here we emit every option and let it choose.
    """
    derived: list[DerivedBool] = []
    subject = _subject(attribute)
    for index, option in enumerate(attribute.labels):
        template = rng.choice(MEMBERSHIP_TEMPLATES)
        instructions = template.format(option=option, subject=subject)
        false_label, true_label = _labels(rng)
        derived.append(DerivedBool(
            question=BoolQuestion(instructions=instructions,
                                  false_label=false_label, true_label=true_label),
            label=int(index == gold_index),
            source_attribute=attribute.name,
            kind="membership",
            schema_id=f"{attribute.name}::is::{option}",
        ))
    return derived


def derive_from_score(
    attribute: Attribute, gold_index: int, rng: random.Random
) -> list[DerivedBool]:
    """K-1 threshold questions from one K-level scale: "is it at least level k?".

    Thresholds are the natural boolean reading of an ordinal scale, and unlike
    membership questions they are *not* rare-positive: the positive rate at threshold
    k is P(y >= k), which sweeps from near 1 at the bottom to near 0 at the top. Taken
    together across k they are close to balanced by construction.
    """
    derived: list[DerivedBool] = []
    subject = _subject(attribute)
    for k in range(1, len(attribute.labels)):
        level = attribute.labels[k]
        template = rng.choice(THRESHOLD_TEMPLATES)
        instructions = template.format(level=level, subject=subject)
        false_label, true_label = _labels(rng)
        derived.append(DerivedBool(
            question=BoolQuestion(instructions=instructions,
                                  false_label=false_label, true_label=true_label),
            label=int(gold_index >= k),
            source_attribute=attribute.name,
            kind="threshold",
            schema_id=f"{attribute.name}::atleast::{k}",
        ))
    return derived


def derive_from_bool(
    attribute: Attribute, gold_index: int, rng: random.Random
) -> list[DerivedBool]:
    """The original question plus a template-built negation with the label flipped."""
    false_label, true_label = _labels(rng)
    original = DerivedBool(
        question=BoolQuestion(instructions=attribute.instructions,
                              false_label=false_label, true_label=true_label),
        label=gold_index,
        source_attribute=attribute.name,
        kind="original",
        schema_id=f"{attribute.name}::original",
    )
    neg_false, neg_true = _labels(rng)
    negated = DerivedBool(
        question=BoolQuestion(instructions=_negate(attribute.instructions, rng),
                              false_label=neg_false, true_label=neg_true),
        label=1 - gold_index,
        source_attribute=attribute.name,
        kind="negation",
        schema_id=f"{attribute.name}::negated",
    )
    return [original, negated]


def derive_all(
    domain: Domain, labels: dict[str, int], rng: random.Random
) -> list[DerivedBool]:
    """Every boolean question derivable from one document's gold labels."""
    derived: list[DerivedBool] = []
    for attribute in domain.attributes:
        gold = labels[attribute.name]
        if attribute.kind == "choice":
            derived.extend(derive_from_choice(attribute, gold, rng))
        elif attribute.kind == "score":
            derived.extend(derive_from_score(attribute, gold, rng))
        elif attribute.kind == "bool":
            derived.extend(derive_from_bool(attribute, gold, rng))
    return derived


def schema_catalog(rng: random.Random | None = None) -> dict[str, dict[str, object]]:
    """Every derivable boolean schema id, for counting and for the leak check."""
    from sokudan.data.builders.synthetic import CATALOG

    rng = rng or random.Random(0)
    catalog: dict[str, dict[str, object]] = {}
    for domain in CATALOG:
        for attribute in domain.attributes:
            if attribute.kind == "choice":
                for option in attribute.labels:
                    catalog[f"{attribute.name}::is::{option}"] = {
                        "domain": domain.name, "kind": "membership",
                        "source": attribute.name, "option": option,
                    }
            elif attribute.kind == "score":
                for k in range(1, len(attribute.labels)):
                    catalog[f"{attribute.name}::atleast::{k}"] = {
                        "domain": domain.name, "kind": "threshold",
                        "source": attribute.name, "level": attribute.labels[k],
                    }
            elif attribute.kind == "bool":
                catalog[f"{attribute.name}::original"] = {
                    "domain": domain.name, "kind": "original", "source": attribute.name,
                }
                catalog[f"{attribute.name}::negated"] = {
                    "domain": domain.name, "kind": "negation", "source": attribute.name,
                }
    return catalog
