"""Run every bench_ja baseline under identical conditions (SOKUDAN_SPEC.md §4.2, §9).

    uv run python scripts/run_baseline_ja.py --bench data/bench_ja.jsonl --out runs/baseline_ja

Same 300 items, same three questions, same probability floor, one metrics module.
TypeSafe Jev is deliberately absent: see docs/baseline_ja.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

# Imported first so that .env is loaded before huggingface_hub / laya read os.environ
# (HF_HUB_DISABLE_SYMLINKS matters on Windows without developer mode).
import sokudan.config  # noqa: F401
from sokudan.calibration.metrics import binary_to_probs
from sokudan.calibration.temperature import cross_fit_temperature
from sokudan.eval.baselines import (
    PROB_FLOOR,
    BaselineOutput,
    LayaBaseline,
    LocalLLMClassifierBaseline,
    MajorityClassBaseline,
    RandomBaseline,
    floor_and_renormalise,
)
from sokudan.eval.bench_ja import DEPARTMENTS, URGENCY_LEVELS, BenchItem
from sokudan.eval.report import (
    gold_arrays,
    markdown_accuracy_table,
    markdown_latency_table,
    save_reliability_diagram,
    score_baseline,
)


def load_bench(path: Path) -> list[BenchItem]:
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(BenchItem(**json.loads(line)))
    return items


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["device_name"] = torch.cuda.get_device_name(0)
            env["device_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:  # pragma: no cover
        env["torch"] = f"unavailable: {exc}"
    try:
        import laya

        env["laya"] = getattr(laya, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover
        env["laya"] = f"unavailable: {exc}"
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=str, default="data/bench_ja.jsonl")
    parser.add_argument("--out", type=str, default="runs/baseline_ja")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N items")
    parser.add_argument("--skip-llm", action="store_true", help="skip the slow LLM baseline")
    parser.add_argument("--sokudan-checkpoint", default=None,
                        help="evaluate a trained sokudan checkpoint alongside the baselines")
    parser.add_argument("--sokudan-temperatures", default=None,
                        help="temperatures.json from scripts/calibrate.py (Stage 2)")
    args = parser.parse_args()

    items = load_bench(Path(args.bench))
    if args.limit:
        items = items[: args.limit]
    print(f"bench_ja: {len(items)} items from {args.bench}")

    gold = gold_arrays(items)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    baselines: list[Any] = [
        LayaBaseline("convaiinnovations/laya-multilingual", "laya-multilingual (ja)"),
        LayaBaseline("convaiinnovations/laya", "laya (英語版モデルに日本語入力)"),
        MajorityClassBaseline(),
        RandomBaseline(),
    ]
    if not args.skip_llm:
        baselines.append(LocalLLMClassifierBaseline())

    if args.sokudan_checkpoint:
        from sokudan.eval.sokudan_baseline import SokudanBaseline

        # Uncalibrated first, then the Stage 2 temperatures, so the effect of
        # calibration on the *same* checkpoint is visible rather than bundled in.
        baselines.append(SokudanBaseline(args.sokudan_checkpoint, "sokudan-ja-310m"))
        if args.sokudan_temperatures:
            blob = json.loads(Path(args.sokudan_temperatures).read_text(encoding="utf-8"))
            parsed = {}
            for key, value in blob["temperatures"].items():
                kind, _, count = key.partition("/")
                parsed[(kind, int(count))] = float(value)
            baselines.append(SokudanBaseline(
                args.sokudan_checkpoint, "sokudan-ja-310m + 温度較正",
                temperatures=parsed,
            ))

    rows: list[dict[str, Any]] = []
    choice_probs_by_name: dict[str, np.ndarray] = {}
    outputs: list[BaselineOutput] = []

    def report(row: dict[str, Any]) -> None:
        print(f"  choice acc {row['choice']['accuracy']:.3f}  ECE {row['choice']['ece']:.3f}"
              f"  | score RPS {row['score']['rps']:.3f}"
              f"  | bool acc {row['bool']['accuracy']:.3f}", flush=True)

    for baseline in baselines:
        print(f"\n--- {baseline.name} ---", flush=True)
        started = time.time()
        output = baseline.run(items)
        outputs.append(output)
        row = score_baseline(output, gold)
        row["wall_clock_s"] = round(time.time() - started, 1)
        rows.append(row)
        choice_probs_by_name[baseline.name] = floor_and_renormalise(output.choice_probs)
        report(row)

    # Laya's own model card says it ships uncalibrated and asks for one temperature per
    # (question type, option count) to be refitted on held-out data. Scoring it only as
    # shipped would be scoring a configuration its authors tell you not to trust, so the
    # cross-fitted row is added as the fair comparison. Fitting is out-of-fold, so no
    # item ever helps calibrate itself.
    for output in list(outputs):
        if "laya" not in output.name.lower():
            continue
        name = f"{output.name} + 温度較正（2-fold交差適合）"
        print(f"\n--- {name} ---", flush=True)
        choice_fit = cross_fit_temperature(
            floor_and_renormalise(output.choice_probs), gold["choice"])
        score_fit = cross_fit_temperature(
            floor_and_renormalise(output.score_probs), gold["score"])
        bool_fit = cross_fit_temperature(
            floor_and_renormalise(binary_to_probs(output.bool_p_true)), gold["bool"])
        derived = BaselineOutput(
            name=name,
            choice_probs=choice_fit.probs,
            score_probs=score_fit.probs,
            bool_p_true=bool_fit.probs[:, 1],
            per_item_latency_s=list(output.per_item_latency_s),
            parse_failures=output.parse_failures,
            parse_attempts=output.parse_attempts,
            notes={
                "derived_from": output.name,
                "description": "post-hoc temperature scaling, fitted out-of-fold",
                "temperatures_choice": [round(t, 4) for t in choice_fit.temperatures],
                "temperatures_score": [round(t, 4) for t in score_fit.temperatures],
                "temperatures_bool": [round(t, 4) for t in bool_fit.temperatures],
            },
        )
        row = score_baseline(derived, gold)
        row["wall_clock_s"] = 0.0
        rows.append(row)
        choice_probs_by_name[name] = derived.choice_probs
        report(row)


    # Persist the probabilities so a figure can be redrawn without re-running any model.
    np.savez_compressed(
        out_dir / "choice_probs.npz",
        gold=gold["choice"],
        **{f"p{i}": v for i, v in enumerate(choice_probs_by_name.values())},
        names=np.array(list(choice_probs_by_name), dtype=object),
    )

    png = save_reliability_diagram(
        choice_probs_by_name,
        gold["choice"],
        out_dir / "reliability_choice.png",
        title=f"bench_ja: choice（部署ルーティング4択）reliability diagram  n={len(items)}",
    )
    print(f"\nreliability diagram -> {png}")

    payload = {
        "environment": environment(),
        "bench": {
            "path": str(args.bench),
            "n_items": len(items),
            "departments": list(DEPARTMENTS),
            "urgency_levels": URGENCY_LEVELS,
            "gold_counts": {
                "department": {
                    k: int((gold["choice"] == i).sum()) for i, k in enumerate(DEPARTMENTS)
                },
                "urgency": {
                    name: int((gold["score"] == i).sum())
                    for i, name in enumerate(URGENCY_LEVELS)
                },
                "churn_true": int(gold["bool"].sum()),
                "churn_false": int(len(items) - gold["bool"].sum()),
            },
        },
        "probability_floor": PROB_FLOOR,
        "jev": "not measured; see docs/baseline_ja.md",
        "results": rows,
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    tables = (
        "### 精度・較正\n\n" + markdown_accuracy_table(rows)
        + "\n\n### レイテンシ\n\n" + markdown_latency_table(rows) + "\n"
    )
    (out_dir / "tables.md").write_text(tables, encoding="utf-8")

    print(f"results -> {results_path}")
    print(f"tables  -> {out_dir / 'tables.md'}")
    print("\n" + tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
