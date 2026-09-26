"""Condition F: shuffle the option order **per item**, not per condition.

    uv run python scripts/probe_shuffled_positions.py \
        --model laya:convaiinnovations/laya-multilingual --device cpu

Conditions A-E in `docs/baseline_ja.md` each hold one option order fixed across all
300 items. That leaves position and label confounded within a condition: when the
first slot is never chosen, "the model refuses slot 1" and "the model refuses whatever
word happens to sit in slot 1" predict the same table, and only the comparison across
conditions separates them.

Shuffling per item breaks the confound inside a single run. Every label lands in every
slot across the 300 items, so:

  * a **slot** histogram that is flat means there is no positional preference;
  * a **label** histogram that is flat means there is no lexical preference;
  * a slot histogram with a hole at index 0 and a label histogram that is *not*
    degenerate is positional, full stop.

The shuffle is seeded per item from a fixed base, so the permutation for item i is
the same on every run and the table is reproducible.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import sokudan.config  # noqa: F401
from sokudan.eval.baselines import floor_and_renormalise
from sokudan.eval.bench_en import URGENCY_LEVELS as EN_URGENCY
from sokudan.eval.bench_en import BenchItem as EnBenchItem
from sokudan.eval.bench_ja import URGENCY_LEVELS
from sokudan.eval.bench_ja import BenchItem as JaBenchItem

INSTRUCTIONS = "この依頼の緊急度は"
EN_INSTRUCTIONS = "The urgency of this request is"

# The two benchmarks do not share a row schema -- bench_en carries a `verified` field --
# so loading one with the other's dataclass raises TypeError rather than mis-parsing.
LANGS = {
    "ja": (JaBenchItem, URGENCY_LEVELS, INSTRUCTIONS),
    "en": (EnBenchItem, EN_URGENCY, EN_INSTRUCTIONS),
}


def load_bench(path: Path, item_cls) -> list:
    return [
        item_cls(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="laya:convaiinnovations/laya-multilingual")
    parser.add_argument("--bench", default="data/bench_ja.jsonl")
    parser.add_argument("--lang", choices=("ja", "en"), default="ja",
                        help="which benchmark schema and option set --bench holds")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--out", default="runs/position_bias_ja_shuffled.json")
    args = parser.parse_args(argv)

    item_cls, URGENCY_LEVELS, INSTRUCTIONS = LANGS[args.lang]
    items = load_bench(Path(args.bench), item_cls)
    kind, _, target = args.model.partition(":")

    if kind == "laya":
        import laya

        agent = laya.load(target, device=args.device)

        def score(state: str, criteria: list[str]) -> np.ndarray:
            questions = {"urgency": {"type": "score", "instructions": INSTRUCTIONS,
                                     "criteria": criteria}}
            answer = agent.predict({"body": state}, questions)["answers"]["urgency"]
            probs = answer.get("probabilities", {})
            row = np.array([float(probs.get(str(k), 0.0)) for k in range(len(criteria))])
            if row.sum() <= 0:
                row = np.full(len(criteria), 1.0 / len(criteria))
            return row / row.sum()
    elif kind == "sokudan":
        import sokudan as pkg
        from sokudan.schema.question import ScoreQuestion

        agent = pkg.load(target, device=args.device)

        def score(state: str, criteria: list[str]) -> np.ndarray:
            question = ScoreQuestion(instructions=INSTRUCTIONS, criteria=criteria)
            answer = agent.predict({"body": state},
                                   {"urgency": question})["answers"]["urgency"]
            probs = answer["probabilities"]
            return np.array([float(probs[str(k)]) for k in range(len(criteria))])
    else:
        raise SystemExit(f"unknown model spec {args.model!r}")

    slot_counts: Counter = Counter()
    label_counts: Counter = Counter()
    correct = 0
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        rng = random.Random(args.seed + index)
        order = list(range(len(URGENCY_LEVELS)))
        rng.shuffle(order)
        criteria = [URGENCY_LEVELS[i] for i in order]

        probs = floor_and_renormalise(score(item.state, criteria).reshape(1, -1))[0]
        slot = int(probs.argmax())
        canonical = order[slot]

        slot_counts[slot] += 1
        label_counts[URGENCY_LEVELS[canonical]] += 1
        correct += int(canonical == item.urgency)
        rows.append({
            "item_id": item.item_id, "presented": criteria,
            "argmax_slot": slot, "chosen_label": URGENCY_LEVELS[canonical],
            "gold": URGENCY_LEVELS[item.urgency],
        })

    n = len(items)
    n_slots = len(URGENCY_LEVELS)
    # How often each label was *offered* in slot 0, so the slot-0 count has a
    # denominator rather than being read against an assumed uniform.
    offered_in_slot0: Counter = Counter()
    for index in range(n):
        rng = random.Random(args.seed + index)
        order = list(range(len(URGENCY_LEVELS)))
        rng.shuffle(order)
        offered_in_slot0[URGENCY_LEVELS[order[0]]] += 1

    print(f"== 条件 F: 件ごとに選択肢順をランダムシャッフル  n={n}  model={target} ==\n")
    print(f"第1スロットが選ばれた件数: {slot_counts[0]} / {n} "
          f"({slot_counts[0]/n:.1%})   一様なら {n/n_slots:.0f} (33.3%)")
    print(f"\nスロット別 argmax: {[slot_counts[i] for i in range(n_slots)]}")
    print(f"ラベル別 選択件数 : "
          f"{ {label: label_counts[label] for label in URGENCY_LEVELS} }")
    print(f"第1スロットに置かれた回数: "
          f"{ {label: offered_in_slot0[label] for label in URGENCY_LEVELS} }")
    print(f"\naccuracy（正準ラベルに戻したあと）: {correct/n:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": target, "bench": args.bench, "n_items": n, "seed": args.seed,
        "condition": "F: per-item random option order",
        "slot_argmax_counts": [slot_counts[i] for i in range(n_slots)],
        "first_slot_count": slot_counts[0],
        "first_slot_share": slot_counts[0] / n,
        "label_counts": {label: label_counts[label] for label in URGENCY_LEVELS},
        "offered_in_slot0": {label: offered_in_slot0[label] for label in URGENCY_LEVELS},
        "accuracy": correct / n,
        "per_item": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
