"""Per-class breakdown of one baseline on bench_ja, for the write-up.

    uv run python scripts/analyze_baseline_ja.py --model convaiinnovations/laya-multilingual

Prints the confusion matrix for `choice`, the predicted-class distribution (a model
that collapses onto one label looks accurate on a skewed set but is not), and the
confidence/accuracy gap per question type.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sokudan.config  # noqa: F401  -- loads .env before huggingface_hub reads it
from sokudan.eval.baselines import LayaBaseline, floor_and_renormalise
from sokudan.eval.bench_ja import DEPARTMENTS, URGENCY_LEVELS, BenchItem
from sokudan.eval.report import gold_arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="convaiinnovations/laya-multilingual")
    parser.add_argument("--bench", default="data/bench_ja.jsonl")
    parser.add_argument("--out", default="runs/baseline_ja/breakdown.json")
    args = parser.parse_args()

    items = [
        BenchItem(**json.loads(line))
        for line in Path(args.bench).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold = gold_arrays(items)
    output = LayaBaseline(args.model, args.model).run(items)

    dept_keys = list(DEPARTMENTS)
    choice = floor_and_renormalise(output.choice_probs)
    pred = choice.argmax(axis=1)

    print(f"\n== {args.model} : choice 混同行列 (n={len(items)}) ==")
    print("gold \ pred      " + "".join(f"{k:>8}" for k in dept_keys) + "    計")
    confusion = np.zeros((len(dept_keys), len(dept_keys)), dtype=int)
    for g, p in zip(gold["choice"], pred, strict=True):
        confusion[g, p] += 1
    for i, k in enumerate(dept_keys):
        print(f"{k:<12}" + "".join(f"{v:>8}" for v in confusion[i]) + f"{confusion[i].sum():>7}")
    print("計          " + "".join(f"{v:>8}" for v in confusion.sum(axis=0))
          + f"{confusion.sum():>7}")

    print("\n== 予測クラスの偏り ==")
    for i, k in enumerate(dept_keys):
        gold_share = (gold["choice"] == i).mean()
        pred_share = (pred == i).mean()
        print(f"  {k:<6} gold {gold_share:.3f}   pred {pred_share:.3f}")

    print("\n== 確信度 vs 正答率 ==")
    breakdown = {}
    for label, probs, gold_vec in (
        ("choice", choice, gold["choice"]),
        ("score", floor_and_renormalise(output.score_probs), gold["score"]),
    ):
        conf = probs.max(axis=1).mean()
        acc = float((probs.argmax(axis=1) == gold_vec).mean())
        print(f"  {label:<7} mean confidence {conf:.3f}   accuracy {acc:.3f}   gap {conf-acc:+.3f}")
        breakdown[label] = {"mean_confidence": float(conf), "accuracy": acc}

    p_true = np.asarray(output.bool_p_true)
    bool_acc = float(((p_true >= 0.5).astype(int) == gold["bool"]).mean())
    print(f"  bool    mean P(true) {p_true.mean():.3f}   gold rate {gold['bool'].mean():.3f}"
          f"   accuracy {bool_acc:.3f}")
    breakdown["bool"] = {
        "mean_p_true": float(p_true.mean()),
        "gold_true_rate": float(gold["bool"].mean()),
        "accuracy": bool_acc,
    }

    print("\n== score の混同（順序尺度）==")
    score_pred = floor_and_renormalise(output.score_probs).argmax(axis=1)
    for i, name in enumerate(URGENCY_LEVELS):
        row = [int(((gold["score"] == i) & (score_pred == j)).sum())
               for j in range(len(URGENCY_LEVELS))]
        print(f"  gold {name:<12} -> pred {row}   (計 {sum(row)})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "n_items": len(items),
                "choice_confusion": confusion.tolist(),
                "choice_labels": dept_keys,
                "choice_pred_share": {k: float((pred == i).mean())
                                      for i, k in enumerate(dept_keys)},
                "choice_gold_share": {k: float((gold["choice"] == i).mean())
                                      for i, k in enumerate(dept_keys)},
                "per_type": breakdown,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
