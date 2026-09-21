"""Aggregate `bench_ja` across seeds and check the v0.1 gate.

    uv run python scripts/report_bench_seeds.py \
        --runs runs/bench_seed0 runs/bench_seed1 runs/bench_seed2

Reports mean ± standard deviation over seeds, uncalibrated and calibrated, and
decides the gate on the *mean*:

    bool acc   > 0.703   (the majority-class baseline; Day 1 missed it at 0.623)
    bool AUROC > 0.65    (Day 1: 0.513 -- the ranking was the real failure)
    score RPS  < 0.197   (maintain; Day 1 reached 0.168)

`mean P(true)` is reported beside `bool acc` on purpose. Day 1's s0 scored 0.683 on
accuracy by answering "false" almost every time, against a gold rate of 0.297 -- an
accuracy that looked competitive and meant nothing. Accuracy without the base rate
beside it is the number that hid that, and `docs/benchmarks.md` §8 says not to
print one without the other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

GATES = {"bool_acc": (0.703, "gt"), "bool_auroc": (0.65, "gt"), "score_rps": (0.197, "lt")}

METRICS: list[tuple[str, tuple[str, str], str]] = [
    ("choice acc", ("choice", "accuracy"), "{:.3f}"),
    ("choice ECE", ("choice", "ece"), "{:.3f}"),
    ("score RPS↓", ("score", "rps"), "{:.3f}"),
    ("score acc", ("score", "accuracy"), "{:.3f}"),
    ("bool acc", ("bool", "accuracy"), "{:.3f}"),
    ("bool AUROC", ("bool", "auroc"), "{:.3f}"),
    ("bool mean P(true)", ("bool", "mean_p_true"), "{:.3f}"),
]


def read_row(run_dir: Path, calibrated: bool) -> dict[tuple[str, str], float]:
    blob = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    for entry in blob["results"]:
        name = str(entry.get("name", ""))
        if "sokudan" not in name.lower():
            continue
        if ("較正" in name) != calibrated:
            continue
        out: dict[tuple[str, str], float] = {}
        for _label, key, _fmt in METRICS:
            kind, metric = key
            value = (entry.get(kind) or {}).get(metric)
            if value is not None:
                out[key] = float(value)
        return out
    raise SystemExit(f"{run_dir}: no {'calibrated ' if calibrated else ''}sokudan row")


def summarise(rows: list[dict[tuple[str, str], float]]) -> dict[tuple[str, str], tuple]:
    out = {}
    for _label, key, _fmt in METRICS:
        values = [r[key] for r in rows if key in r]
        if not values:
            out[key] = (None, None, [])
        elif len(values) == 1:
            out[key] = (values[0], None, values)
        else:
            out[key] = (mean(values), stdev(values), values)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="runs/bench_v01_seeds.json")
    args = parser.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    plain = summarise([read_row(d, False) for d in run_dirs])
    calibrated = summarise([read_row(d, True) for d in run_dirs])

    def cell(entry: tuple, fmt: str) -> str:
        value, deviation, _ = entry
        if value is None:
            return "n/a"
        if deviation is None:
            return fmt.format(value)
        return f"{fmt.format(value)} ± {fmt.format(deviation)}"

    print(f"{len(run_dirs)} シード: {[d.name for d in run_dirs]}\n")
    print(f"{'指標':20s} {'較正前':>20s} {'較正後':>20s}")
    for label, key, fmt in METRICS:
        print(f"{label:20s} {cell(plain[key], fmt):>20s} {cell(calibrated[key], fmt):>20s}")

    print("\n各シードの生値（較正前）")
    for label, key, fmt in METRICS:
        values = plain[key][2]
        print(f"  {label:20s} {[fmt.format(v) for v in values]}")

    print("\nv0.1 ゲート（3 シード平均、較正前）")
    verdicts = {}
    for name, (threshold, direction) in GATES.items():
        kind, metric = {"bool_acc": ("bool", "accuracy"),
                        "bool_auroc": ("bool", "auroc"),
                        "score_rps": ("score", "rps")}[name]
        value = plain[(kind, metric)][0]
        ok = value is not None and (value > threshold if direction == "gt" else value < threshold)
        verdicts[name] = ok
        symbol = ">" if direction == "gt" else "<"
        print(f"  {name:12s} {value:.4f} {symbol} {threshold}   {'PASS' if ok else 'FAIL'}")

    summary = {
        "runs": [str(d) for d in run_dirs],
        "n_seeds": len(run_dirs),
        "uncalibrated": {f"{k[0]}/{k[1]}": {"mean": v[0], "sd": v[1], "values": v[2]}
                         for k, v in plain.items()},
        "calibrated": {f"{k[0]}/{k[1]}": {"mean": v[0], "sd": v[1], "values": v[2]}
                       for k, v in calibrated.items()},
        "gates": {k: {"passed": v, "threshold": GATES[k][0]} for k, v in verdicts.items()},
        "all_passed": all(verdicts.values()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")

    print("\n通過" if summary["all_passed"] else "\n未達")
    return 0 if summary["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
