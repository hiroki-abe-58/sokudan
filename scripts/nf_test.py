"""docs/noise_floor.md §4 (Phase 4): input order, 8 vs 8, both at batch 24.

    uv run python scripts/nf_test.py

Test 1: M2, "sf > v0.1", one-sided exact permutation test (all 12,870 splits).
Test 2: M5, "sf < v0.1", one-sided exact permutation test.
Holm over the two, alpha 0.05. Runs only when both conditions have 8 seeds; otherwise
it prints the descriptive numbers and says the test was not run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.seed8_stats import exact_permutation_p

ALPHA = 0.05
V01 = ["runs/io/eval_v01.json"] + [f"runs/nf/eval_v01_seed{s}.json" for s in range(3, 8)]
SF = (["runs/io/eval_seed0.json", "runs/io/eval_seed12.json"]
      + [f"runs/nf/eval_io_sf_seed{s}.json" for s in range(3, 8)])


def holm(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm step-down: adjusted p and reject decision per test."""
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    out, running, stop = {}, 0.0, False
    for rank, name in enumerate(order):
        adjusted = min(1.0, max(running, (m - rank) * pvalues[name]))
        running = adjusted
        reject = not stop and pvalues[name] <= alpha / (m - rank)
        stop = stop or not reject
        out[name] = {"p": float(pvalues[name]), "p_holm": float(adjusted),
                     "reject": bool(reject)}
    return out


def load(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        if Path(p).exists():
            out += json.loads(Path(p).read_text(encoding="utf-8"))
    return out


def main() -> int:
    v01, sf = load(V01), load(SF)
    m2 = {"v01": [e["M2"]["auroc"] for e in v01], "sf": [e["M2"]["auroc"] for e in sf]}
    m5 = {"v01": [e["M1_by_attribute"]["implies_declining"]["auroc"] for e in v01],
          "sf": [e["M1_by_attribute"]["implies_declining"]["auroc"] for e in sf]}
    report = {"n": {"v01": len(v01), "sf": len(sf)}, "M2": m2, "M5": m5,
              "means": {k: {g: float(np.mean(v)) for g, v in d.items()}
                        for k, d in (("M2", m2), ("M5", m5))}}
    if len(v01) == 8 and len(sf) == 8:
        d1, p1, n1 = exact_permutation_p(m2["sf"], m2["v01"])     # sf > v0.1
        d2, p2, n2 = exact_permutation_p(m5["v01"], m5["sf"])     # sf < v0.1
        report["tests"] = {
            "test1 M2 sf > v01": {"statistic": d1, "splits": n1},
            "test2 M5 sf < v01": {"statistic": d2, "splits": n2},
        }
        decided = holm({"test1 M2 sf > v01": p1, "test2 M5 sf < v01": p2})
        for name, result in decided.items():
            report["tests"][name].update(result)
    else:
        report["tests"] = "not run: fewer than 8 seeds in a condition"
    Path("runs/nf/phase4_tests.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
