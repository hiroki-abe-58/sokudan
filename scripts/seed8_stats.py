"""Tests and intervals pre-registered in docs/seed8.md.

    uv run python scripts/seed8_stats.py

- **Superiority (the one primary test)**: M2, statistic mean(long12) - mean(base12),
  one-sided exact permutation test over all C(16, 8) = 12,870 splits, alpha 0.05.
- **Non-inferiority** (intersection-union, every one must pass): shift long12 by its
  margin in the favourable direction (+0.005 for M1, choice acc and bool acc, +0.01 for
  M5 and score acc; -0.005 for score RPS, where lower is better), then the same
  one-sided exact permutation test against base12, alpha 0.05.
- **Description only**: bootstrap 95% intervals of each mean difference (seeds resampled
  with replacement within each condition, 10,000 draws, numpy seed `BOOT_SEED`), and the
  batch-composition comparisons against the shipped v0.1 (batch 24, seeds 0-2).

The p-value counts the observed split itself, so the smallest possible is 1 / 12,870.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

ALPHA = 0.05
BOOT_SEED = 20260925
BOOT_N = 10_000
# (metric, margin applied to long12, direction: +1 higher is better, -1 lower is better)
NON_INFERIORITY = [("M1", 0.005, 1), ("M5", 0.01, 1), ("choice_accuracy", 0.005, 1),
                   ("score_accuracy", 0.01, 1), ("bool_accuracy", 0.005, 1),
                   ("score_rps", 0.005, -1)]
METRICS = ["M2", "M1", "M5", "choice_accuracy", "score_rps", "score_accuracy",
           "bool_accuracy", "M3 (ref)", "M4 (ref)", "400-599 (ref)", "600-799 (ref)"]


def exact_permutation_p(a: list[float], b: list[float]) -> tuple[float, float, int]:
    """One-sided p for mean(a) - mean(b) being this large, over every relabelling.

    Returns (observed statistic, p, number of splits).
    """
    pooled = np.array(list(a) + list(b), dtype=float)
    n, k = len(pooled), len(a)
    observed = float(np.mean(a) - np.mean(b))
    total = pooled.sum()
    count = splits = 0
    for idx in itertools.combinations(range(n), k):
        s = pooled[list(idx)].sum()
        stat = s / k - (total - s) / (n - k)
        # A small tolerance so a split equal to the observed one is not lost to rounding.
        count += stat >= observed - 1e-12
        splits += 1
    return observed, count / splits, splits


def bootstrap_ci(a: list[float], b: list[float], rng: np.random.Generator) -> tuple[float, float]:
    a_arr, b_arr = np.array(a), np.array(b)
    ia = rng.integers(0, len(a_arr), size=(BOOT_N, len(a_arr)))
    ib = rng.integers(0, len(b_arr), size=(BOOT_N, len(b_arr)))
    diffs = a_arr[ia].mean(axis=1) - b_arr[ib].mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def metrics(entry: dict) -> dict[str, float]:
    g = entry["guardrails"]
    return {
        "M2": entry["M2"]["auroc"], "M1": entry["M1"]["auroc"],
        "M5": entry["M1_by_attribute"]["implies_declining"]["auroc"],
        "choice_accuracy": g["choice_accuracy"], "score_rps": g["score_rps"],
        "score_accuracy": g["score_accuracy"], "bool_accuracy": g["bool_accuracy"],
        "M3 (ref)": entry["M3"]["accuracy"], "M4 (ref)": entry["M4"]["accuracy"],
        "400-599 (ref)": entry["bins_pooled"]["400-600"]["auroc"],
        "600-799 (ref)": entry["bins_pooled"]["600-800"]["auroc"],
    }


def load(paths: list[str]) -> list[dict[str, float]]:
    out = []
    for p in paths:
        path = Path(p)
        if path.exists():
            out += [metrics(e) for e in json.loads(path.read_text(encoding="utf-8"))]
    return out


def paths() -> dict[str, list[str]]:
    long12 = [f"runs/l2x2/eval_qf_long_seed{s}.json" for s in range(3)]
    long12 += [f"runs/seed8/eval_long12_seed{s}.json" for s in range(3, 8)]
    return {
        "base12": [f"runs/seed8/eval_base12_seed{s}.json" for s in range(8)],
        "long12": long12,
        "v01_batch24": ["runs/io/eval_v01.json"],
    }


def main() -> int:
    data = {name: load(p) for name, p in paths().items()}
    col = {name: {m: [s[m] for s in seeds] for m in METRICS} for name, seeds in data.items()}
    report: dict = {"n_seeds": {k: len(v) for k, v in data.items()}}
    for name, c in col.items():
        report[name] = {m: {"values": v, "mean": float(np.mean(v)) if v else None,
                            "min": min(v) if v else None, "max": max(v) if v else None}
                        for m, v in c.items()}

    complete = len(data["base12"]) == 8 and len(data["long12"]) == 8
    report["complete_8v8"] = complete
    rng = np.random.default_rng(BOOT_SEED)
    if data["base12"] and data["long12"]:
        report["bootstrap_ci_long12_minus_base12"] = {
            m: {"diff": float(np.mean(col["long12"][m]) - np.mean(col["base12"][m])),
                "ci95": bootstrap_ci(col["long12"][m], col["base12"][m], rng)}
            for m in METRICS
        }
    if complete:
        obs, p, splits = exact_permutation_p(col["long12"]["M2"], col["base12"]["M2"])
        report["primary"] = {"metric": "M2", "diff": obs, "p": p, "splits": splits,
                             "pass": p < ALPHA}
        ni = {}
        for metric, margin, direction in NON_INFERIORITY:
            longs, bases = col["long12"][metric], col["base12"][metric]
            if direction > 0:
                shifted = [x + margin for x in longs]
                stat, p, _ = exact_permutation_p(shifted, bases)
            else:
                shifted = [x - margin for x in longs]
                stat, p, _ = exact_permutation_p(bases, shifted)
            ni[metric] = {"margin": margin * (-direction), "diff": float(np.mean(longs) -
                                                                         np.mean(bases)),
                          "shifted_stat": stat, "p": p, "pass": p < ALPHA}
        report["non_inferiority"] = ni
        all_ni = all(v["pass"] for v in ni.values())
        if report["primary"]["pass"] and all_ni:
            verdict = "採用候補"
        elif report["primary"]["pass"]:
            verdict = "トレードオフ"
        else:
            verdict = "長さの効果は未検出"
        report["verdict"] = verdict
    else:
        report["verdict"] = "TBD（8 対 8 がそろっていないため検定しない）"

    # Descriptive: batch composition (base12 vs v0.1 batch 24, same seeds 0-2) and the
    # shipped model (v0.1, 3 seeds) against long12 (8 seeds).
    if len(data["base12"]) >= 3:
        report["base12_vs_v01_seeds0_2"] = {
            m: {"base12": col["base12"][m][:3], "v01": col["v01_batch24"][m],
                "mean_diff": float(np.mean(col["base12"][m][:3]) - np.mean(col["v01_batch24"][m]))}
            for m in METRICS}
    if data["long12"]:
        report["long12_vs_v01"] = {
            m: float(np.mean(col["long12"][m]) - np.mean(col["v01_batch24"][m])) for m in METRICS}
    if data["base12"]:
        allm2 = sorted(col["base12"]["M2"] + col["v01_batch24"]["M2"])
        report["v01_seed1_M2_position"] = {
            "value": col["v01_batch24"]["M2"][1],
            "rank_among_base12_and_v01": allm2.index(col["v01_batch24"]["M2"][1]) + 1,
            "of": len(allm2),
            "base12_below": sum(v < col["v01_batch24"]["M2"][1] for v in col["base12"]["M2"]),
            "base12_n": len(col["base12"]["M2"]),
        }
    out = Path("runs/seed8/stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("base12", "long12",
                                                                   "v01_batch24")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
