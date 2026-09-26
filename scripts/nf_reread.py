"""docs/noise_floor.md §2 (Phase 3): re-read every checkpoint against v0.1's 8 seeds.

    uv run python scripts/nf_reread.py

Reads the `scripts/eval_local_attention.py` outputs already on disk (the same script and
code produce the same numbers; one checkpoint is re-evaluated to confirm it) and adds:

- **M1m**: the unweighted mean of the per-attribute held-out AUROCs (attributes with
  n >= 30), next to M1, which pools the rows and so weights each attribute by its
  row count;
- every attribute's AUROC and its share of M1's rows;
- v0.1's 8-seed mean, SD (ddof 1), min and max, and where each other condition's mean
  sits: (mean - v0.1 mean) / v0.1 SD, and how many of v0.1's 8 seeds it exceeds.

Writes `runs/nf/reread.json`. Description only: no past verdict is changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COLLAPSE = 0.60  # ends_with_question held-out AUROC below this is a "collapse" (§1.3)
SOURCES = {
    "v01 (batch 24, qf)": ["runs/io/eval_v01.json"]
    + [f"runs/nf/eval_v01_seed{s}.json" for s in range(3, 8)],
    "io_sf (batch 24, sf)": ["runs/io/eval_seed0.json", "runs/io/eval_seed12.json"]
    + [f"runs/nf/eval_io_sf_seed{s}.json" for s in range(3, 8)],
    "sf-long (2x2, batch 12x2)": [f"runs/l2x2/eval_sf_long_seed{s}.json" for s in range(3)],
    "long12 (qf-long, batch 12x2)": [f"runs/l2x2/eval_qf_long_seed{s}.json" for s in range(3)]
    + [f"runs/seed8/eval_long12_seed{s}.json" for s in range(3, 8)],
    "base12 (batch 12x2, pre-fix)": ["runs/seed8/eval_base12_seed0.json"],
    "base12 (batch 12x2, fixed)": ["runs/nf/eval_base12fix_seed0.json"],
    "la1024 (batch 24, window 1024)": ["runs/la/eval_la1024_seed0.json"],
}
METRICS = ["M1", "M1m", "M2", "M5", "choice_accuracy", "score_rps", "score_accuracy",
           "bool_accuracy", "ends_with_question AUROC"]


def row(entry: dict) -> dict:
    attrs = entry["M1_by_attribute"]
    usable = {a: v for a, v in attrs.items() if v["n"] >= 30 and "auroc" in v}
    g = entry["guardrails"]
    return {
        "checkpoint": entry["checkpoint"],
        "M1": entry["M1"]["auroc"],
        "M1m": float(np.mean([v["auroc"] for v in usable.values()])),
        "M2": entry["M2"]["auroc"],
        "M5": attrs["implies_declining"]["auroc"],
        "choice_accuracy": g["choice_accuracy"], "score_rps": g["score_rps"],
        "score_accuracy": g["score_accuracy"], "bool_accuracy": g["bool_accuracy"],
        "ends_with_question AUROC": attrs["ends_with_question"]["auroc"],
        "by_attribute": {a: {"n": v["n"], "auroc": v.get("auroc")} for a, v in attrs.items()},
    }


def load(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        if Path(p).exists():
            out += [row(e) for e in json.loads(Path(p).read_text(encoding="utf-8"))]
    return out


def main() -> int:
    data = {name: load(paths) for name, paths in SOURCES.items()}
    report: dict = {"seeds": {k: len(v) for k, v in data.items()}, "conditions": {}}
    base = data["v01 (batch 24, qf)"]
    first = base[0]["by_attribute"]
    total = sum(v["n"] for v in first.values())
    report["row_share"] = {a: {"n": v["n"], "share": v["n"] / total} for a, v in first.items()}
    ref = {}
    for m in METRICS:
        v = np.array([r[m] for r in base])
        ref[m] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                  "min": float(v.min()), "max": float(v.max()), "values": v.tolist()}
    report["v01_distribution"] = ref
    report["v01_collapses"] = sum(r["ends_with_question AUROC"] < COLLAPSE for r in base)
    for name, rows in data.items():
        if not rows:
            continue
        cond = {"n_seeds": len(rows),
                "collapses": sum(r["ends_with_question AUROC"] < COLLAPSE for r in rows),
                "per_seed": rows, "position": {}}
        for m in METRICS:
            mean = float(np.mean([r[m] for r in rows]))
            sd = ref[m]["sd"]
            cond["position"][m] = {
                "mean": mean,
                "z_vs_v01": (mean - ref[m]["mean"]) / sd if sd else None,
                "v01_seeds_below": int(sum(x < mean for x in ref[m]["values"])),
            }
        report["conditions"][name] = cond
    Path("runs/nf/reread.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps({"seeds": report["seeds"], "v01_collapses": report["v01_collapses"],
                      "row_share": report["row_share"]}, ensure_ascii=False, indent=1))
    for m in METRICS:
        r = ref[m]
        sd = f"{r['sd']:.4f}" if r["sd"] is not None else "-"
        print(f"v01 {m:26s} mean {r['mean']:.4f} sd {sd} min {r['min']:.4f} max {r['max']:.4f}")
    n_ref = len(base)
    for name, cond in report["conditions"].items():
        pos = cond["position"]
        cells = []
        for m in METRICS:
            p = pos[m]
            if p["z_vs_v01"] is None:
                cells.append(f"{m} {p['mean']:.4f}")
            else:
                cells.append(f"{m} {p['mean']:.4f} (z {p['z_vs_v01']:+.2f}, "
                             f">{p['v01_seeds_below']}/{n_ref})")
        print(name, cond["n_seeds"], "collapses", cond["collapses"], " | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
