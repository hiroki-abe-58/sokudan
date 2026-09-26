"""Diagnostic 1: does the model use the state at all?

    uv run python scripts/diagnose_state_use.py --checkpoint runs/s0/model.pt

Replaces each item's state with (a) the empty string and (b) a different document
from the same benchmark, keeping the questions and the gold labels fixed. If the
three metrics barely move, the model is answering from the schema alone and the
cross-attention path is not carrying information -- which would explain a
bench_ja result that sits at the majority baseline.

The shuffled condition is the sharper of the two: an empty state is out of
distribution and could degrade for unrelated reasons, whereas a real Japanese
business document that simply belongs to a *different* item is perfectly in
distribution and merely wrong.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

import sokudan.config  # noqa: F401
from sokudan.calibration.metrics import auroc, binary_to_probs, rps
from sokudan.eval.baselines import floor_and_renormalise
from sokudan.eval.bench_ja import BenchItem
from sokudan.eval.report import gold_arrays
from sokudan.eval.sokudan_baseline import SokudanBaseline


def metrics_for(output: Any, gold: dict[str, np.ndarray]) -> dict[str, float]:
    choice = floor_and_renormalise(output.choice_probs)
    score = floor_and_renormalise(output.score_probs)
    p_true = np.asarray(output.bool_p_true, dtype=np.float64)
    boolean = floor_and_renormalise(binary_to_probs(p_true))
    entry = {
        "choice_acc": float((choice.argmax(1) == gold["choice"]).mean()),
        "score_rps": rps(score, gold["score"]),
        "score_acc": float((score.argmax(1) == gold["score"]).mean()),
        "bool_acc": float((boolean.argmax(1) == gold["bool"]).mean()),
        "mean_p_true": float(p_true.mean()),
    }
    if 0 < gold["bool"].sum() < len(gold["bool"]):
        entry["bool_auroc"] = auroc(p_true, gold["bool"])
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/s0/model.pt")
    parser.add_argument("--bench", default="data/bench_ja.jsonl")
    parser.add_argument("--out", default="runs/diagnostics/state_use.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    items = [
        BenchItem(**json.loads(line))
        for line in Path(args.bench).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold = gold_arrays(items)

    # A derangement: no item keeps its own state.
    rng = random.Random(args.seed)
    order = list(range(len(items)))
    while True:
        rng.shuffle(order)
        if all(i != j for i, j in enumerate(order)):
            break

    conditions = {
        "本来の state": items,
        "state を空文字に": [
            BenchItem(**{**item.to_json(), "state": ""}) for item in items
        ],
        "state を別文書に差し替え": [
            BenchItem(**{**item.to_json(), "state": items[order[i]].state})
            for i, item in enumerate(items)
        ],
    }

    baseline = SokudanBaseline(args.checkpoint, "sokudan")
    results: dict[str, dict[str, float]] = {}
    for name, condition_items in conditions.items():
        results[name] = metrics_for(baseline.run(condition_items), gold)

    keys = ["choice_acc", "score_rps", "score_acc", "bool_acc", "bool_auroc", "mean_p_true"]
    print(f"\n== 診断1: state 差し替え  n={len(items)} ==\n")
    print(f"{'条件':<26} " + " ".join(f"{k:>12}" for k in keys))
    print("-" * (26 + 13 * len(keys)))
    for name, entry in results.items():
        print(f"{name:<26} " + " ".join(
            f"{entry.get(k, float('nan')):>12.4f}" for k in keys))

    base = results["本来の state"]
    print("\n本来の state との差:")
    verdicts = []
    for name, entry in results.items():
        if name == "本来の state":
            continue
        deltas = {k: entry.get(k, float("nan")) - base.get(k, float("nan")) for k in keys}
        print(f"  {name:<26} " + " ".join(f"{deltas[k]:>+12.4f}" for k in keys))
        moved = abs(deltas["choice_acc"]) + abs(deltas["score_acc"]) + abs(deltas["bool_acc"])
        verdicts.append((name, moved))

    print()
    shuffled_move = dict(verdicts)["state を別文書に差し替え"]
    if shuffled_move < 0.03:
        verdict = ("STATE IGNORED: swapping in a different document moves the three "
                   f"accuracies by {shuffled_move:.4f} in total. The model is answering "
                   "from the schema alone.")
    elif shuffled_move < 0.10:
        verdict = (f"WEAK STATE USE: total accuracy movement {shuffled_move:.4f}. "
                   "The state matters, but far less than it should.")
    else:
        verdict = (f"STATE IS USED: total accuracy movement {shuffled_move:.4f}. "
                   "The failure is elsewhere.")
    print(f"判定: {verdict}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"checkpoint": args.checkpoint, "n_items": len(items),
                    "conditions": results, "verdict": verdict},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
