"""`bench_en` for a sokudan checkpoint (docs/release_candidate.md Phase 5, description only).

    uv run python scripts/run_bench_en_sokudan.py --checkpoint runs/release_candidate/model.pt \
        --out runs/release_candidate/bench_en

`scripts/run_baseline_en.py` has no sokudan row, and `SokudanBaseline` asks the `bench_ja`
questions. This runs the same `SokudanBaseline` with only the three questions swapped for
`bench_en.bench_questions()` (same option counts: 4 departments, 3 urgency levels, one
boolean), and scores it with the same `score_baseline` and `bench_en` gold arrays that
`run_baseline_en.py` uses. Nothing else about the inference or scoring path changes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sokudan.config  # noqa: F401
import sokudan.eval.sokudan_baseline as sokudan_baseline
from scripts.run_baseline_en import gold_arrays, load_bench
from sokudan.eval import bench_en, bench_ja
from sokudan.eval.report import score_baseline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bench", default="data/bench_en.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    assert len(bench_en.DEPARTMENTS) == len(bench_ja.DEPARTMENTS)
    assert len(bench_en.URGENCY_LEVELS) == len(bench_ja.URGENCY_LEVELS)
    sokudan_baseline.bench_questions = bench_en.bench_questions

    items = load_bench(Path(args.bench))
    gold = gold_arrays(items)
    baseline = sokudan_baseline.SokudanBaseline(args.checkpoint, "sokudan (bench_en)")
    started = time.time()
    row = score_baseline(baseline.run(items), gold)
    row["wall_clock_s"] = round(time.time() - started, 1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "bench": args.bench,
            "n_items": len(items), "checkpoint": args.checkpoint, "results": [row]}
    path = out_dir / "results.json"
    path.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"choice acc {row['choice']['accuracy']:.4f}  score RPS {row['score']['rps']:.4f}"
          f"  score acc {row['score']['accuracy']:.4f}  bool acc {row['bool']['accuracy']:.4f}"
          f"  bool AUROC {row['bool'].get('auroc', float('nan')):.4f}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
