"""Measure the baselines on `bench_en`, under the same rules as `bench_ja`.

    uv run python scripts/run_baseline_en.py --bench data/bench_en.jsonl --out runs/baseline_en

Deliberately a separate script rather than a `--lang` switch on
`scripts/run_baseline_ja.py`. That script produced the published Japanese numbers;
threading a second language through it would risk changing them silently, and the
whole value of `bench_en` is that the Japanese measurement stays exactly as it was.

The scoring path is shared: the same probability floor, the same metric functions
from `sokudan.calibration.metrics`, the same `score_baseline`. Only the item type and
the question strings differ.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

import sokudan.config  # noqa: F401
from sokudan.eval.baselines import PROB_FLOOR, BaselineOutput
from sokudan.eval.bench_en import DEPARTMENTS, URGENCY_LEVELS, BenchItem, bench_questions
from sokudan.eval.report import score_baseline


def load_bench(path: Path) -> list[BenchItem]:
    return [
        BenchItem(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def gold_arrays(items: list[BenchItem]) -> dict[str, np.ndarray]:
    keys = list(DEPARTMENTS)
    return {
        "choice": np.array([keys.index(i.department) for i in items]),
        "score": np.array([i.urgency for i in items]),
        "bool": np.array([int(i.churn) for i in items]),
    }


class LayaBaselineEn:
    """A Laya checkpoint answering the `bench_en` questions verbatim."""

    def __init__(self, model_id: str, label: str) -> None:
        self.model_id = model_id
        self.name = label
        self._agent: Any | None = None

    def run(self, items: list[BenchItem]) -> BaselineOutput:
        import laya

        if self._agent is None:
            self._agent = laya.load(self.model_id)
        questions = bench_questions()
        keys = list(DEPARTMENTS)

        choice_rows, score_rows, bool_vals, latencies = [], [], [], []
        malformed = 0
        for item in items:
            started = time.perf_counter()
            result = self._agent.predict({"body": item.state}, questions)
            latencies.append(time.perf_counter() - started)
            answers = result.get("answers", {})

            dept = answers.get("department", {}).get("probabilities", {})
            row = np.array([float(dept.get(k, 0.0)) for k in keys])
            if row.sum() <= 0:
                malformed += 1
                row = np.full(len(keys), 1.0 / len(keys))
            choice_rows.append(row / row.sum())

            urgency = answers.get("urgency", {}).get("probabilities", {})
            srow = np.array([
                float(urgency.get(str(k), 0.0)) for k in range(len(URGENCY_LEVELS))
            ])
            if srow.sum() <= 0:
                malformed += 1
                srow = np.full(len(URGENCY_LEVELS), 1.0 / len(URGENCY_LEVELS))
            score_rows.append(srow / srow.sum())

            noul = answers.get("churn", {}).get("noul")
            if noul is None:
                malformed += 1
                noul = 0.5
            bool_vals.append(float(noul))

        return BaselineOutput(
            name=self.name,
            choice_probs=np.vstack(choice_rows),
            score_probs=np.vstack(score_rows),
            bool_p_true=np.array(bool_vals),
            per_item_latency_s=latencies,
            parse_failures=malformed,
            parse_attempts=len(items) * 3,
            notes={"model_id": self.model_id, "bench": "bench_en"},
        )


class MajorityEn:
    name = "majority class"

    def run(self, items: list[BenchItem]) -> BaselineOutput:
        gold = gold_arrays(items)
        n = len(items)

        def constant(values: np.ndarray, width: int) -> np.ndarray:
            counts = np.bincount(values, minlength=width)
            row = np.zeros(width)
            row[int(counts.argmax())] = 1.0
            return np.tile(row, (n, 1))

        rate = float(gold["bool"].mean())
        return BaselineOutput(
            name=self.name,
            choice_probs=constant(gold["choice"], len(DEPARTMENTS)),
            score_probs=constant(gold["score"], len(URGENCY_LEVELS)),
            bool_p_true=np.full(n, 1.0 if rate > 0.5 else 0.0),
            per_item_latency_s=[], parse_failures=0, parse_attempts=0,
            notes={"description": "the most frequent gold label, every time"},
        )


class RandomEn:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def run(self, items: list[BenchItem]) -> BaselineOutput:
        n = len(items)

        def uniform(width: int) -> np.ndarray:
            # Jitter, or argmax always lands on index 0 and "random" silently
            # becomes "the first option" -- the bug docs/day1 found.
            rows = np.full((n, width), 1.0 / width)
            return rows + self.rng.normal(0, 1e-3, size=rows.shape)

        return BaselineOutput(
            name=self.name,
            choice_probs=np.abs(uniform(len(DEPARTMENTS))),
            score_probs=np.abs(uniform(len(URGENCY_LEVELS))),
            bool_p_true=self.rng.random(n),
            per_item_latency_s=[], parse_failures=0, parse_attempts=0,
            notes={"description": "uniform with jitter"},
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="data/bench_en.jsonl")
    parser.add_argument("--out", default="runs/baseline_en")
    parser.add_argument("--skip-laya", action="store_true")
    args = parser.parse_args()

    items = load_bench(Path(args.bench))
    gold = gold_arrays(items)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    baselines: list[Any] = []
    if not args.skip_laya:
        baselines += [
            LayaBaselineEn("convaiinnovations/laya-multilingual", "laya-multilingual (en)"),
            LayaBaselineEn("convaiinnovations/laya", "laya (english model)"),
        ]
    baselines += [MajorityEn(), RandomEn()]

    rows = []
    for baseline in baselines:
        print(f"\n== {baseline.name} ==", flush=True)
        row = score_baseline(baseline.run(items), gold)
        rows.append(row)
        print(f"  choice acc {row['choice']['accuracy']:.3f}  ECE {row['choice']['ece']:.3f}"
              f"  | score RPS {row['score']['rps']:.3f}  acc {row['score']['accuracy']:.3f}"
              f"  | bool acc {row['bool']['accuracy']:.3f}"
              f"  AUROC {row['bool'].get('auroc', float('nan')):.3f}")

    blob = {
        "environment": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "bench": {
            "path": args.bench, "n_items": len(items),
            "departments": list(DEPARTMENTS),
            "urgency_levels": list(URGENCY_LEVELS),
            "gold_counts": {
                "department": {k: int((gold["choice"] == i).sum())
                               for i, k in enumerate(DEPARTMENTS)},
                "urgency": {k: int((gold["score"] == i).sum())
                            for i, k in enumerate(URGENCY_LEVELS)},
                "churn_true": int(gold["bool"].sum()),
                "churn_false": int(len(items) - gold["bool"].sum()),
            },
        },
        "probability_floor": PROB_FLOOR,
        "results": rows,
    }
    path = out_dir / "results.json"
    path.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
