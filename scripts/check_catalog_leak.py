"""Static `bench_ja` contamination check over the catalogues themselves.

    uv run python scripts/check_catalog_leak.py

The existing checks run on *rows*, after generation, and they look at two fields:
the question's `instructions` and its option strings (`build_train_data.leakage_check`,
`rebuild_train_data.leak_check`). That was enough when every boolean question came
from a `choice` or `score` label. It is not enough now.

The Day 2 attributes carry four more places a `bench_ja` schema could enter, none of
which ever becomes a row field:

    * `behaviour` -- the instruction the *generator* is given
    * the attribute definitions themselves (true/false behaviour, banned phrases)
    * the three paraphrases of every question
    * the negation template

A leak in any of those would be invisible to a row-level check and would still put a
`bench_ja` schema into training, which invalidates every generalisation number
reported afterwards. So this runs over the catalogue objects, before generation,
and it fails the build rather than reporting a count.

The ban covers questions, not bodies. `docs/benchmarks.md` §5 §3.2 has the
reasoning: 2.7% of the existing training documents already use this vocabulary with
no label attached, and stripping it would leave a model that has never seen those
tokens in context -- which is not protection from `bench_ja`, it is a handicap on it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from typing import Any

from sokudan.data import intent_attributes as ia
from sokudan.data.builders.synthetic import CATALOG, STYLES
from sokudan.data.schema_aug import BOOL_LABEL_PAIRS, DISTRACTOR_LABELS, INSTRUCTION_SUFFIXES
from sokudan.eval.bench_ja import DEPARTMENTS, URGENCY_LEVELS

# Anything that reads as the bench_ja boolean, however it is spelled. Wider than
# `rebuild_train_data.BENCH_BOOL_TERMS`: 解除 and 退会 are contract termination under
# a different word, and a question built from either would be the bench question.
BENCH_TERMS = [
    "解約", "契約終了", "契約の終了", "契約解除", "解除", "退会", "脱退",
    "乗り換え", "乗換", "他社", "他のサービス", "競合", "別の会社", "よその会社",
]

BENCH_INSTRUCTIONS = {
    "この問い合わせはどの部署が担当すべきか",
    "この依頼の緊急度は",
    "送信者は解約・契約終了を示唆しているか",
}

BENCH_LABELS = set(DEPARTMENTS) | set(URGENCY_LEVELS)


def walk_catalog() -> Iterator[tuple[str, str, str]]:
    """Yield `(path, field, text)` for every author-written string in the catalogues.

    `field` is one of `instructions` / `label` / `free` and decides which checks
    apply: exact-match checks only mean something against the field they mirror.
    """
    for domain in CATALOG:
        root = f"domain:{domain.name}"
        yield root, "free", domain.document
        for index, context in enumerate(domain.contexts):
            yield f"{root}.contexts[{index}]", "free", context
        for word in domain.banned_words:
            yield f"{root}.banned_words", "free", word
        for attribute in domain.attributes:
            base = f"{root}.{attribute.name}"
            yield f"{base}.instructions", "instructions", attribute.instructions
            for label in attribute.labels:
                yield f"{base}.labels", "label", label
            for key, text in attribute.descriptions.items():
                yield f"{base}.descriptions[{key}]", "free", text
            for index, text in enumerate(attribute.behaviour):
                yield f"{base}.behaviour[{index}]", "free", text
            for key, forms in attribute.surface_forms.items():
                for form in forms:
                    yield f"{base}.surface_forms[{key}]", "label", form

    for attribute in ia.ATTRIBUTES:
        base = f"intent:{attribute.id}"
        for index, form in enumerate(attribute.forms):
            yield f"{base}.forms[{index}]", "instructions", form
        yield f"{base}.negation_form", "instructions", attribute.negation_form
        yield f"{base}.true_behaviour", "free", attribute.true_behaviour
        yield f"{base}.false_behaviour", "free", attribute.false_behaviour
        for phrase in attribute.banned_phrases:
            yield f"{base}.banned_phrases", "free", phrase

    for index, style in enumerate(STYLES):
        yield f"shared:STYLES[{index}]", "free", style
    for index, suffix in enumerate(INSTRUCTION_SUFFIXES):
        yield f"shared:INSTRUCTION_SUFFIXES[{index}]", "free", suffix
    for false_label, true_label in BOOL_LABEL_PAIRS:
        yield "shared:BOOL_LABEL_PAIRS", "label", false_label
        yield "shared:BOOL_LABEL_PAIRS", "label", true_label
    for label, description in DISTRACTOR_LABELS:
        yield "shared:DISTRACTOR_LABELS", "label", label
        yield "shared:DISTRACTOR_LABELS", "free", description


def check() -> dict[str, Any]:
    term_hits: list[dict[str, str]] = []
    instruction_hits: list[dict[str, str]] = []
    label_hits: list[dict[str, str]] = []
    scanned = 0

    for path, field, text in walk_catalog():
        scanned += 1
        for term in BENCH_TERMS:
            if term in text:
                term_hits.append({"path": path, "term": term, "text": text})
        if field == "instructions" and text in BENCH_INSTRUCTIONS:
            instruction_hits.append({"path": path, "text": text})
        if field == "label" and text in BENCH_LABELS:
            label_hits.append({"path": path, "text": text})

    return {
        "strings_scanned": scanned,
        "bench_term_hits": term_hits,
        "bench_instruction_collisions": instruction_hits,
        "bench_label_collisions": label_hits,
        "clean": not (term_hits or instruction_hits or label_hits),
    }


def check_intent_catalog() -> dict[str, Any]:
    """Internal consistency of the intent catalogue, independent of `bench_ja`."""
    problems: list[str] = []

    seen_forms: dict[str, str] = {}
    for attribute in ia.ATTRIBUTES:
        for form in [*attribute.forms, attribute.negation_form]:
            if form in seen_forms and seen_forms[form] != attribute.id:
                problems.append(
                    f"question text shared by {seen_forms[form]} and {attribute.id}: {form}"
                )
            seen_forms[form] = attribute.id

    if len(ia.HELD_OUT - set(ia.BY_ID)) > 0:
        problems.append(f"held-out names nothing: {sorted(ia.HELD_OUT - set(ia.BY_ID))}")

    held_tiers = {ia.BY_ID[name].tier for name in ia.HELD_OUT}
    if held_tiers != set(ia.TIERS):
        # The Day 2 gate is three conditions, one per tier. A held-out set missing a
        # tier makes one of them unevaluable, so this is a hard failure, not a note.
        problems.append(
            "held-out must span every tier for the three-condition gate; "
            f"covers {sorted(held_tiers)}"
        )

    for left, right in ia.EXCLUSIVE_PAIRS:
        for name in (left, right):
            if name not in ia.BY_ID:
                problems.append(f"exclusive pair names nothing: {name}")

    # A trainable attribute whose every conflict partner is also trainable is fine;
    # what would break selection is a *held-out* attribute conflicting with so much
    # that a validation document cannot be filled.
    forced = [ia.BY_ID[name] for name in sorted(ia.HELD_OUT)]
    blocked = set()
    for attribute in forced:
        blocked |= ia.conflicts_with(attribute.id)
        blocked.add(attribute.id)
    available = [a for a in ia.TRAINABLE if a.id not in blocked]
    if len(available) < 3:
        problems.append(
            f"only {len(available)} trainable attributes are compatible with the held-out set"
        )

    return {"problems": problems, "clean": not problems, "val_fillable_from": len(available)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=str, default="runs/catalog_leak.json")
    args = parser.parse_args()

    leak = check()
    consistency = check_intent_catalog()

    print(f"scanned {leak['strings_scanned']} catalogue strings "
          f"({len(CATALOG)} domains, {len(ia.ATTRIBUTES)} intent attributes)")

    for key, label in (
        ("bench_term_hits", "bench_ja 語彙"),
        ("bench_instruction_collisions", "bench_ja 質問文"),
        ("bench_label_collisions", "bench_ja 選択肢"),
    ):
        hits = leak[key]
        if hits:
            print(f"\n{label}: {len(hits)} 件")
            for hit in hits:
                term = hit.get("term", "")
                print(f"  {hit['path']}  {term}  -> {hit['text']}")

    if consistency["problems"]:
        print("\nintent catalogue problems:")
        for problem in consistency["problems"]:
            print(f"  {problem}")

    from pathlib import Path
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"leak": leak, "consistency": consistency,
             "intent_catalog": ia.catalog_summary()},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {out}")

    if leak["clean"] and consistency["clean"]:
        print("CLEAN")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
