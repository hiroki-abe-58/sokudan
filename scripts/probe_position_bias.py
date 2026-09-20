"""Position-bias probe, for any model that answers a `score` question.

    uv run python scripts/probe_position_bias.py --model laya:convaiinnovations/laya-multilingual
    uv run python scripts/probe_position_bias.py --model sokudan:runs/s0/model.pt

`docs/baseline_ja.md` §6.2 measured `laya-multilingual` putting 0, 0, 1, 1 and 0
predictions on the *first presented option* out of 300, across five schema variants.
That is the failure `sokudan`'s dynamic-K ordinal head is meant to avoid, so the same
probe has to run against `sokudan` before any such claim is made.

Counts are reported in **presented order, with no remapping**, because the remapping
is exactly where an analysis bug would hide. A separate canonical-order pass supplies
accuracy and Spearman, which need the labels aligned.

Nothing about this script is specific to the two models it knows how to load; adding
a third means adding one adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

import sokudan.config  # noqa: F401
from sokudan.calibration.metrics import ordinal_mae, rps, spearman
from sokudan.eval.baselines import floor_and_renormalise
from sokudan.eval.bench_ja import URGENCY_LEVELS, BenchItem
from sokudan.schema.question import ScoreQuestion

INSTRUCTIONS = "この依頼の緊急度は"

# The same five schemas used on Laya, so the two are directly comparable.
# `reverse` says how to map the presented order back onto the canonical scale.
VARIANTS: list[dict[str, Any]] = [
    {"name": "A 原文", "criteria": list(URGENCY_LEVELS), "reverse": False},
    {"name": "B 逆順", "criteria": list(reversed(URGENCY_LEVELS)), "reverse": True},
    {"name": "C 言い換え（低/中/高）", "criteria": ["低", "中", "高"], "reverse": False},
    {"name": "D 言い換えの逆順", "criteria": ["高", "中", "低"], "reverse": True},
    {"name": "E 4段階", "reverse": False,
     "criteria": ["全く急がない", "急がない", "早めに", "業務が止まっている"],
     "gold_map": {0: 0, 1: 1, 2: 3}},
]


class ScoreModel(Protocol):
    name: str

    def score(self, items: list[BenchItem], criteria: list[str]) -> np.ndarray:
        """Return `(N, len(criteria))` probabilities in the *presented* order."""
        ...


class LayaScorer:
    def __init__(self, model_id: str) -> None:
        import laya

        self.name = model_id
        self._agent = laya.load(model_id)

    def score(self, items: list[BenchItem], criteria: list[str]) -> np.ndarray:
        questions = {"urgency": {"type": "score", "instructions": INSTRUCTIONS,
                                 "criteria": criteria}}
        rows = []
        for item in items:
            answer = self._agent.predict({"body": item.state}, questions)["answers"]["urgency"]
            probs = answer.get("probabilities", {})
            row = np.array([float(probs.get(str(k), 0.0)) for k in range(len(criteria))])
            rows.append(row if row.sum() > 0 else np.full(len(criteria), 1.0 / len(criteria)))
        return floor_and_renormalise(np.vstack(rows))


class SokudanScorer:
    def __init__(self, checkpoint: str, temperatures: str | None = None) -> None:
        import sokudan as pkg

        self.name = f"sokudan({checkpoint})"
        self._agent = pkg.load(checkpoint, temperatures=temperatures)

    def score(self, items: list[BenchItem], criteria: list[str]) -> np.ndarray:
        question = ScoreQuestion(instructions=INSTRUCTIONS, criteria=criteria)
        rows = []
        for item in items:
            answer = self._agent.predict({"body": item.state},
                                         {"urgency": question})["answers"]["urgency"]
            probs = answer["probabilities"]
            rows.append(np.array([float(probs[str(k)]) for k in range(len(criteria))]))
        return floor_and_renormalise(np.vstack(rows))


def build(spec: str, temperatures: str | None) -> ScoreModel:
    kind, _, target = spec.partition(":")
    if kind == "laya":
        return LayaScorer(target)
    if kind == "sokudan":
        return SokudanScorer(target, temperatures)
    raise ValueError(f"unknown model spec {spec!r}; use laya:<id> or sokudan:<checkpoint>")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="laya:<id> or sokudan:<checkpoint>")
    parser.add_argument("--temperatures", default=None)
    parser.add_argument("--bench", default="data/bench_ja.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    items = [
        BenchItem(**json.loads(line))
        for line in Path(args.bench).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold = np.array([item.urgency for item in items])
    model = build(args.model, args.temperatures)

    print(f"== 位置バイアス検査: {model.name}  n={len(items)} ==\n")
    print(f"{'条件':<24} {'提示順の argmax 件数':<28} {'第1選択肢':>9} {'acc':>7} "
          f"{'RPS':>7} {'MAE':>7} {'spearman':>9}")
    print("-" * 100)

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        criteria = variant["criteria"]
        probs = model.score(items, criteria)
        presented_counts = np.bincount(probs.argmax(axis=1), minlength=len(criteria))

        # Canonical order, for the label-aligned metrics.
        canonical = probs[:, ::-1] if variant["reverse"] else probs
        gold_map = variant.get("gold_map")
        if gold_map:
            mapped_gold = np.array([gold_map[int(g)] for g in gold])
        else:
            mapped_gold = gold

        accuracy = float((canonical.argmax(axis=1) == mapped_gold).mean())
        levels = np.arange(canonical.shape[1])
        correlation = spearman(canonical @ levels, mapped_gold)

        entry = {
            "name": variant["name"],
            "criteria": criteria,
            "presented_argmax_counts": presented_counts.tolist(),
            "first_option_count": int(presented_counts[0]),
            "accuracy": accuracy,
            "rps": rps(canonical, mapped_gold),
            "mae_argmax": ordinal_mae(canonical, mapped_gold),
            "spearman_expectation_vs_gold": correlation,
        }
        rows.append(entry)
        print(f"{variant['name']:<24} {str(presented_counts.tolist()):<28} "
              f"{presented_counts[0]:>9} {accuracy:>7.3f} {entry['rps']:>7.3f} "
              f"{entry['mae_argmax']:>7.3f} {correlation:>9.3f}")

    first_counts = [r["first_option_count"] for r in rows]
    n = len(items)
    dead = sum(1 for c in first_counts if c <= 0.01 * n)
    share = [c / n for c in first_counts]

    print(f"\n第1選択肢が選ばれた割合: {[f'{s:.1%}' for s in share]}")
    if dead == len(rows):
        verdict = ("the first presented option is never selected under any schema -- "
                   "the same positional failure measured on laya-multilingual")
    elif dead:
        verdict = (f"the first option is effectively dead in {dead}/{len(rows)} schemas -- "
                   "partial positional bias")
    else:
        verdict = ("the first presented option is selected under every schema -- "
                   "no dead-slot positional bias")
    print(f"判定: {verdict}")

    accuracies = [r["accuracy"] for r in rows]
    spread = max(accuracies) - min(accuracies)
    print(f"スキーマ間の accuracy 振れ幅: {min(accuracies):.3f}〜{max(accuracies):.3f} "
          f"(幅 {spread:.3f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"model": model.name, "n_items": n, "variants": rows,
                    "first_option_share": share, "verdict": verdict,
                    "accuracy_spread": spread}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
