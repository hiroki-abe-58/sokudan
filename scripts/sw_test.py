"""docs/sandwich.md: H1, H2 with Holm, plus the reference-only numbers.

    uv run python scripts/sw_test.py

H1: M2, sandwich > v0.1. H2: M5, sandwich > sf. One-sided exact permutation tests,
8 vs 8, Holm over the two, alpha 0.05 (docs/research_protocol.md §3). Reference only:
M5 non-inferiority of sandwich vs v0.1 at margin -0.01 (sandwich + 0.01 > v0.1), and
bootstrap 95% intervals of the mean differences (numpy seed 20260926, 10,000 draws).
Runs the tests only with 8 seeds in every condition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.nf_test import holm
from scripts.seed8_stats import bootstrap_ci, exact_permutation_p

BOOT_SEED = 20260926
SOURCES = {
    "v01": ["runs/io/eval_v01.json"] + [f"runs/nf/eval_v01_seed{s}.json" for s in range(3, 8)],
    "sf": ["runs/io/eval_seed0.json", "runs/io/eval_seed12.json"]
    + [f"runs/nf/eval_io_sf_seed{s}.json" for s in range(3, 8)],
    "sandwich": [f"runs/sw/eval_sandwich_seed{s}.json" for s in range(8)],
}
METRICS = ["M2", "M5", "M1m", "M1", "choice_accuracy", "score_rps", "score_accuracy",
           "bool_accuracy", "ends_with_question"]


def metrics(e: dict) -> dict[str, float]:
    attrs = e["M1_by_attribute"]
    g = e["guardrails"]
    return {
        "M2": e["M2"]["auroc"], "M5": attrs["implies_declining"]["auroc"],
        "M1m": float(np.mean([v["auroc"] for v in attrs.values()
                              if v["n"] >= 30 and "auroc" in v])),
        "M1": e["M1"]["auroc"], "choice_accuracy": g["choice_accuracy"],
        "score_rps": g["score_rps"], "score_accuracy": g["score_accuracy"],
        "bool_accuracy": g["bool_accuracy"],
        "ends_with_question": attrs["ends_with_question"]["auroc"],
    }


def load(paths: list[str]) -> list[dict[str, float]]:
    out = []
    for p in paths:
        if Path(p).exists():
            out += [metrics(e) for e in json.loads(Path(p).read_text(encoding="utf-8"))]
    return out


def main() -> int:
    data = {name: load(paths) for name, paths in SOURCES.items()}
    col = {n: {m: [s[m] for s in seeds] for m in METRICS} for n, seeds in data.items()}
    report: dict = {"n": {k: len(v) for k, v in data.items()}, "per_seed": col,
                    "summary": {n: {m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                        "min": min(v), "max": max(v)}
                                    for m, v in c.items()} for n, c in col.items() if data[n]}}
    if all(len(v) == 8 for v in data.values()):
        d1, p1, _ = exact_permutation_p(col["sandwich"]["M2"], col["v01"]["M2"])
        d2, p2, _ = exact_permutation_p(col["sandwich"]["M5"], col["sf"]["M5"])
        tests = holm({"H1 M2 sandwich > v01": p1, "H2 M5 sandwich > sf": p2})
        tests["H1 M2 sandwich > v01"]["diff"] = d1
        tests["H2 M5 sandwich > sf"]["diff"] = d2
        report["tests"] = tests
        r1 = tests["H1 M2 sandwich > v01"]["reject"]
        r2 = tests["H2 M5 sandwich > sf"]["reject"]
        report["verdict"] = {
            (True, True): "両取りの兆候（採用に向けた評価は別の実験として設計する）",
            (True, False): "長文の効果は保てたが、断りは戻らなかった",
            (False, True): "断りは戻ったが、長文の効果は消えた",
            (False, False): "サンドイッチは閉じる",
        }[(r1, r2)]
        shifted = [x + 0.01 for x in col["sandwich"]["M5"]]
        _, p_ni, _ = exact_permutation_p(shifted, col["v01"]["M5"])
        report["reference_M5_noninferiority_vs_v01"] = {"margin": -0.01, "p": float(p_ni)}
    else:
        report["tests"] = "not run: fewer than 8 seeds in a condition"
        report["verdict"] = "TBD"
    rng = np.random.default_rng(BOOT_SEED)
    if data["sandwich"]:
        report["bootstrap_ci"] = {
            f"sandwich - {other}": {
                m: {"diff": float(np.mean(col["sandwich"][m]) - np.mean(col[other][m])),
                    "ci95": bootstrap_ci(col["sandwich"][m], col[other][m], rng)}
                for m in METRICS}
            for other in ("v01", "sf")}
    out = Path("runs/sw/tests.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_seed"}, ensure_ascii=False,
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
