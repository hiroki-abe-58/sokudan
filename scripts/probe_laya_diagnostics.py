"""Diagnostics behind the Gate B claims in docs/baseline_ja.md.

Two questions the headline table cannot answer on its own:

1. **Is `bool` actually broken, or merely offset?** Accuracy at a 0.5 threshold
   conflates bad ranking with a good ranking read at the wrong cut point. AUROC and
   a rate-matched threshold separate the two. If AUROC is near 0.5 the model is not
   ranking; if it is high, "broken" is the wrong word and the claim must be softened
   to a calibration offset.

2. **Why does `score` never pick the lowest level?** Three possibilities: the ordinal
   head is degenerate, the option *order* is being read positionally, or the specific
   Japanese wording is the problem. Running the same 300 items under three schema
   variants tells them apart.

    uv run python scripts/probe_laya_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import sokudan.config  # noqa: F401  -- loads .env before huggingface_hub reads it
from sokudan.calibration.metrics import (
    auroc,
    expected_level,
    ordinal_mae,
    rate_matched_accuracy,
    rps,
    spearman,
)
from sokudan.eval.baselines import floor_and_renormalise
from sokudan.eval.bench_ja import URGENCY_LEVELS, BenchItem
from sokudan.eval.report import gold_arrays

BASELINE_URGENCY_INSTRUCTIONS = "この依頼の緊急度は"

# Three ways of asking the *same* ordinal question. The gold labels never change;
# only the schema does. Each variant states how to map its own option order back to
# the canonical 0..2 scale, so the comparison stays apples to apples.
SCORE_VARIANTS: dict[str, dict[str, Any]] = {
    "原文（急がない→早めに→業務が止まっている）": {
        "instructions": BASELINE_URGENCY_INSTRUCTIONS,
        "criteria": list(URGENCY_LEVELS),
        "reverse": False,
    },
    "選択肢の順序を逆転": {
        "instructions": BASELINE_URGENCY_INSTRUCTIONS,
        "criteria": list(reversed(URGENCY_LEVELS)),
        "reverse": True,
    },
    "言い換え（低→中→高）": {
        "instructions": BASELINE_URGENCY_INSTRUCTIONS,
        "criteria": ["低", "中", "高"],
        "reverse": False,
    },
}


def load_items(path: Path) -> list[BenchItem]:
    return [
        BenchItem(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_score_variant(agent: Any, items: list[BenchItem], spec: dict[str, Any]) -> np.ndarray:
    """Return (N, 3) probabilities on the canonical 急がない→止まっている ordering."""
    questions = {
        "urgency": {
            "type": "score",
            "instructions": spec["instructions"],
            "criteria": spec["criteria"],
        }
    }
    rows = []
    n_levels = len(spec["criteria"])
    for item in items:
        answer = agent.predict({"body": item.state}, questions)["answers"]["urgency"]
        probs = answer.get("probabilities", {})
        row = np.array([float(probs.get(str(k), 0.0)) for k in range(n_levels)])
        if row.sum() <= 0:
            row = np.full(n_levels, 1.0 / n_levels)
        row = row / row.sum()
        if spec["reverse"]:
            row = row[::-1]  # map the reversed schema back onto the canonical order
        rows.append(row)
    return floor_and_renormalise(np.vstack(rows))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="data/bench_ja.jsonl")
    parser.add_argument("--model", default="convaiinnovations/laya-multilingual")
    parser.add_argument("--out", default="runs/baseline_ja/diagnostics.json")
    args = parser.parse_args()

    import laya

    items = load_items(Path(args.bench))
    gold = gold_arrays(items)
    agent = laya.load(args.model)
    report: dict[str, Any] = {"model": args.model, "n_items": len(items)}

    # ---------------------------------------------------------------- bool
    from sokudan.eval.baselines import LayaBaseline

    base = LayaBaseline(args.model, args.model).run(items)
    p_true = np.asarray(base.bool_p_true, dtype=np.float64)
    gold_bool = gold["bool"]
    gold_rate = float(gold_bool.mean())

    bool_auroc = auroc(p_true, gold_bool)
    acc_half = float(((p_true >= 0.5).astype(int) == gold_bool).mean())
    acc_matched, threshold = rate_matched_accuracy(p_true, gold_bool)
    majority_acc = max(gold_rate, 1 - gold_rate)

    print(f"\n== bool（解約示唆）診断  n={len(items)} ==")
    print(f"  gold 正例率            {gold_rate:.3f}   (多数決精度 {majority_acc:.3f})")
    print(f"  平均 P(true)           {p_true.mean():.3f}")
    print(f"  精度 @ 閾値0.5         {acc_half:.3f}")
    print(f"  精度 @ 正例率一致閾値   {acc_matched:.3f}  (閾値 {threshold:.4f})")
    print(f"  AUROC                  {bool_auroc:.3f}")

    if bool_auroc < 0.60:
        verdict = "ranking is near chance -- 'broken' is supported"
    elif acc_matched <= majority_acc:
        verdict = ("ranking carries signal but rate-matched accuracy still fails to beat "
                   "majority -- weak, not merely offset")
    else:
        verdict = ("ranking carries real signal and beats majority once the threshold is "
                   "matched -- this is a calibration offset, soften the claim")
    print(f"  判定: {verdict}")

    report["bool"] = {
        "gold_positive_rate": gold_rate,
        "majority_accuracy": majority_acc,
        "mean_p_true": float(p_true.mean()),
        "accuracy_at_0.5": acc_half,
        "accuracy_rate_matched": acc_matched,
        "rate_matched_threshold": threshold,
        "auroc": bool_auroc,
        "verdict": verdict,
    }

    # --------------------------------------------------------------- score
    print(f"\n== score（緊急度）スキーマ 3 条件  n={len(items)} ==")
    gold_score = gold["score"]
    score_report: dict[str, Any] = {}
    for name, spec in SCORE_VARIANTS.items():
        probs = run_score_variant(agent, items, spec)
        pred = probs.argmax(axis=1)
        counts = np.bincount(pred, minlength=len(URGENCY_LEVELS))
        expectation = expected_level(probs)
        entry = {
            "criteria": spec["criteria"],
            "reversed": spec["reverse"],
            "pred_counts": counts.tolist(),
            "lowest_level_predicted": int(counts[0]),
            "accuracy": float((pred == gold_score).mean()),
            "rps": rps(probs, gold_score),
            "mae_argmax": ordinal_mae(probs, gold_score),
            "spearman_expectation_vs_gold": spearman(expectation, gold_score),
        }
        score_report[name] = entry
        print(f"\n  --- {name} ---")
        print(f"    選択肢          {spec['criteria']}")
        print(f"    予測件数        {counts.tolist()}  (最下位ラベル {counts[0]} 件)")
        print(f"    accuracy        {entry['accuracy']:.3f}")
        print(f"    RPS             {entry['rps']:.3f}")
        print(f"    MAE(argmax)     {entry['mae_argmax']:.3f}")
        print(f"    spearman(期待値, gold)  {entry['spearman_expectation_vs_gold']:.3f}")

    dead_everywhere = all(v["lowest_level_predicted"] == 0 for v in score_report.values())
    score_verdict = (
        "the lowest level is never selected under any of the three schemas -- "
        "'the ordinal head is not functioning' is supported"
        if dead_everywhere
        else "at least one schema revives the lowest level -- this is schema fragility, "
             "not a dead ordinal head; rewrite the claim accordingly"
    )
    print(f"\n  判定: {score_verdict}")
    report["score"] = {"variants": score_report, "verdict": score_verdict}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
