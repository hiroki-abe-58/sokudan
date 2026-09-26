"""docs/distill.md: H1-H3 with Holm, plus SD ratios and intervals.

    uv run python scripts/distill_test.py

H1: M1m, student > v0.1. H2: M5, student > v0.1. H3: M2, student > v0.1. One-sided exact
permutation tests, 8 vs 8, Holm over the three, alpha 0.05. Description only: the SD
ratio (student / v0.1) of each metric, M1 and the guardrails, bootstrap 95% intervals
(numpy seed 20260926, 10,000 draws). Tests run only with 8 students.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.nf_test import holm
from scripts.seed8_stats import bootstrap_ci, exact_permutation_p
from scripts.sw_test import METRICS, SOURCES, load

BOOT_SEED = 20260926
STUDENT = [f"runs/distill/eval_distill_seed{s}.json" for s in range(8)]
HYPOTHESES = {"H1 M1m student > v01": "M1m", "H2 M5 student > v01": "M5",
              "H3 M2 student > v01": "M2"}


def main() -> int:
    data = {"v01": load(SOURCES["v01"]), "student": load(STUDENT)}
    col = {n: {m: [s[m] for s in seeds] for m in METRICS} for n, seeds in data.items()}
    report: dict = {
        "n": {k: len(v) for k, v in data.items()}, "per_seed": col,
        "summary": {n: {m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                            "min": min(v), "max": max(v)} for m, v in c.items()}
                    for n, c in col.items() if len(data[n]) > 1},
    }
    if len(data["student"]) == 8 and len(data["v01"]) == 8:
        pvalues, diffs = {}, {}
        for name, metric in HYPOTHESES.items():
            d, p, _ = exact_permutation_p(col["student"][metric], col["v01"][metric])
            pvalues[name], diffs[name] = p, d
        tests = holm(pvalues)
        for name in tests:
            tests[name]["diff"] = diffs[name]
        report["tests"] = tests
        rejected = [n for n, t in tests.items() if t["reject"]]
        report["verdict"] = ("蒸留は有望（採用に向けた評価は別途設計する）" if rejected
                             else "この設定の蒸留は閉じる")
        report["rejected"] = rejected
    else:
        report["tests"] = "not run: fewer than 8 seeds"
        report["verdict"] = "TBD"
    if len(data["student"]) > 1:
        s, v = report["summary"]["student"], report["summary"]["v01"]
        report["sd_ratio"] = {m: s[m]["sd"] / v[m]["sd"] for m in METRICS}
        rng = np.random.default_rng(BOOT_SEED)
        report["bootstrap_ci"] = {
            m: {"diff": float(np.mean(col["student"][m]) - np.mean(col["v01"][m])),
                "ci95": bootstrap_ci(col["student"][m], col["v01"][m], rng)} for m in METRICS}
    out = Path("runs/distill/tests.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_seed"}, ensure_ascii=False,
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
