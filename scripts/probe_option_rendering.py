"""laya#131: three option renderings against both labelled benchmarks.

    uv run python scripts/probe_option_rendering.py --device cpu --out runs/option_rendering.json

Runs `laya-multilingual` and `laya` over `bench_ja` (300) and `bench_en` (290) under
conditions A / C / D from `sokudan.eval.option_rendering`, and reports the slot-0 count,
the per-slot argmax, accuracy, RPS and the per-slot mean of the *raw* logits.

`laya` is the English checkpoint and is the control: if the effect is a property of the
rendering it should appear in both, and if it is a property of the multilingual
checkpoint it should not.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

import sokudan.config  # noqa: F401
from sokudan.eval.bench_en import BenchItem as EnBenchItem
from sokudan.eval.bench_ja import BenchItem as JaBenchItem
from sokudan.eval.option_rendering import CONDITIONS, run_condition
from sokudan.eval.position_bias import EN_INSTRUCTIONS, EN_VARIANTS, INSTRUCTIONS, VARIANTS

BENCHES = {
    "bench_ja": {
        "path": "data/bench_ja.jsonl", "item": JaBenchItem,
        "instructions": INSTRUCTIONS, "criteria": VARIANTS[0]["criteria"],
    },
    "bench_en": {
        "path": "data/bench_en.jsonl", "item": EnBenchItem,
        "instructions": EN_INSTRUCTIONS, "criteria": EN_VARIANTS[0]["criteria"],
    },
}


def load_bench(spec: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    items = [
        spec["item"](**json.loads(line))
        for line in Path(spec["path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [i.state for i in items], np.array([i.urgency for i in items])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["convaiinnovations/laya-multilingual",
                                 "convaiinnovations/laya"])
    parser.add_argument("--benches", nargs="+", default=["bench_ja", "bench_en"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="runs/option_rendering.json")
    args = parser.parse_args(argv)

    import laya

    results: dict[str, Any] = {}
    for model_id in args.models:
        agent = laya.load(model_id, device=args.device)
        results[model_id] = {}
        for bench in args.benches:
            spec = BENCHES[bench]
            states, gold = load_bench(spec)
            if args.limit:
                states, gold = states[:args.limit], gold[:args.limit]
            rows = []
            for name, renderer in CONDITIONS:
                result = run_condition(
                    agent, name, renderer, states, gold,
                    spec["instructions"], list(spec["criteria"]),
                )
                rows.append(asdict(result))
                print(f"{model_id:36s} {bench:9s} {name:20s} "
                      f"slot0={result.first_slot_count:4d}/{result.n} "
                      f"argmax={result.slot_argmax_counts} "
                      f"acc={result.accuracy:.3f} rps={result.rps:.4f} "
                      f"logits={result.mean_logit_by_slot}", flush=True)
            results[model_id][bench] = rows
        del agent

    print(f"\n{'model':30s}{'bench':10s}{'条件':22s}{'第1スロット':>12s}"
          f"{'スロット別 argmax':>26s}{'acc':>8s}{'RPS':>8s}{'生ロジット平均':>30s}")
    print("-" * 146)
    for model_id, benches in results.items():
        for bench, rows in benches.items():
            for row in rows:
                logits = " ".join(f"{v:+.3f}" for v in row["mean_logit_by_slot"])
                print(f"{model_id.split('/')[-1]:30s}{bench:10s}{row['name']:22s}"
                      f"{row['first_slot_count']:>7d}/{row['n']:<4d}"
                      f"{str(row['slot_argmax_counts']):>26s}"
                      f"{row['accuracy']:>8.3f}{row['rps']:>8.4f}{logits:>30s}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
