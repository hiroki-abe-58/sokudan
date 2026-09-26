"""Apply docs/length_2x2.md §5-6 mechanically.

    uv run python scripts/l2x2_gate.py breakage runs/l2x2/eval_qf_long_seed0.json
    uv run python scripts/l2x2_gate.py verdict

`breakage` checks one condition's seed 0 against the M1 floor. `verdict` reads all
four conditions, prints every mean and range, the pairwise "差あり / 差なし" calls, the
result type and the final candidate. Double-precision values from the evaluation
output; nothing is rounded before comparing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDITIONS = {
    "qf-short": ["runs/io/eval_v01.json"],
    "sf-short": ["runs/io/eval_seed0.json", "runs/io/eval_seed12.json"],
    "qf-long": [f"runs/l2x2/eval_qf_long_seed{s}.json" for s in range(3)],
    "sf-long": [f"runs/l2x2/eval_sf_long_seed{s}.json" for s in range(3)],
}
SIMPLICITY = ["qf-short", "qf-long", "sf-short", "sf-long"]
M1_FLOOR = 0.854142  # v0.1 min 0.881718 - 2 x range 0.013788 (docs/length_2x2.md §5)
GUARDS = [("choice_accuracy", "higher"), ("score_rps", "lower"),
          ("score_accuracy", "higher"), ("bool_accuracy", "higher")]


def metrics(entry: dict) -> dict[str, float]:
    g = entry["guardrails"]
    return {
        "M1": entry["M1"]["auroc"], "M2": entry["M2"]["auroc"],
        "M5": entry["M1_by_attribute"]["implies_declining"]["auroc"],
        "choice_accuracy": g["choice_accuracy"], "score_rps": g["score_rps"],
        "score_accuracy": g["score_accuracy"], "bool_accuracy": g["bool_accuracy"],
        "M3 (ref)": entry["M3"]["accuracy"], "M4 (ref)": entry["M4"]["accuracy"],
        "400-599 (ref)": entry["bins_pooled"]["400-600"]["auroc"],
        "600-799 (ref)": entry["bins_pooled"]["600-800"]["auroc"],
    }


def load_condition(paths: list[str]) -> list[dict[str, float]] | None:
    entries = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            return None
        entries += json.loads(path.read_text(encoding="utf-8"))
    return [metrics(e) for e in entries] if len(entries) == 3 else None


def summarise(seeds: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for key in seeds[0]:
        v = [s[key] for s in seeds]
        out[key] = {"values": v, "mean": sum(v) / len(v), "min": min(v), "max": max(v),
                    "range": max(v) - min(v)}
    return out


def differs(a: dict, b: dict, key: str = "M2") -> tuple[bool, float]:
    """(差あり?, mean_a - mean_b). 差あり iff |diff| > max(range_a, range_b)."""
    diff = a[key]["mean"] - b[key]["mean"]
    return abs(diff) > max(a[key]["range"], b[key]["range"]), diff


def greater(a: dict, b: dict) -> bool:
    """"a > b が差あり"."""
    sig, diff = differs(a, b)
    return sig and diff > 0


def result_type(s: dict[str, dict]) -> tuple[str, list[str]]:
    """First matching type, and every type whose condition holds."""
    need = ("qf-short", "sf-short", "qf-long", "sf-long")
    holds = []
    if all(k in s for k in ("qf-short", "sf-short", "qf-long")):
        qfl_vs_sfs_sig, qfl_vs_sfs = differs(s["qf-long"], s["sf-short"])
        if greater(s["qf-long"], s["qf-short"]) and (not qfl_vs_sfs_sig or qfl_vs_sfs > 0):
            holds.append("A")
        if (not differs(s["qf-long"], s["qf-short"])[0]
                and greater(s["sf-short"], s["qf-short"])):
            holds.append("B")
    if all(k in s for k in need):
        if (greater(s["qf-long"], s["qf-short"]) and greater(s["sf-long"], s["qf-long"])
                and greater(s["sf-long"], s["sf-short"])):
            holds.append("C")
    return (holds[0] if holds else "D"), holds


def passes_cut(c: dict, base: dict) -> dict[str, bool]:
    checks = {"M1": c["M1"]["mean"] >= base["M1"]["min"],
              "M5": c["M5"]["mean"] >= base["M5"]["min"]}
    for key, direction in GUARDS:
        checks[key] = (c[key]["mean"] <= base[key]["max"] if direction == "lower"
                       else c[key]["mean"] >= base[key]["min"])
    return checks


def candidate(s: dict[str, dict]) -> tuple[str, dict]:
    base = s["qf-short"]
    cuts = {name: passes_cut(c, base) for name, c in s.items()}
    passing = [name for name in SIMPLICITY if name in s and all(cuts[name].values())]
    if passing == ["qf-short"] or not passing:
        return "qf-short (v0.1 維持)", {"cuts": cuts, "passing": passing}
    best = max(passing, key=lambda n: s[n]["M2"]["mean"])
    tied = [n for n in passing if n == best or not differs(s[n], s[best])[0]]
    chosen = min(tied, key=SIMPLICITY.index)
    return chosen, {"cuts": cuts, "passing": passing, "best_M2": best, "tied_with_best": tied}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("breakage")
    b.add_argument("eval_json")
    sub.add_parser("verdict")
    args = parser.parse_args()

    if args.command == "breakage":
        entry = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))[0]
        m1 = entry["M1"]["auroc"]
        print(f"{entry['checkpoint']}: seed 0 M1 {m1:.6f} vs floor {M1_FLOOR:.6f} -> "
              f"{'BROKEN' if m1 < M1_FLOOR else 'not broken'}")
        return 0

    s = {}
    for name, paths in CONDITIONS.items():
        seeds = load_condition(paths)
        if seeds is None:
            print(f"{name}: unavailable (not 3 seeds)")
            continue
        s[name] = summarise(seeds)
    for name, c in s.items():
        print(f"{name}:")
        for key, v in c.items():
            print(f"  {key:16s} " + " / ".join(f"{x:.6f}" for x in v["values"])
                  + f"  mean {v['mean']:.6f} range {v['range']:.6f}")
    names = list(s)
    print("M2 pairwise (差あり iff |mean diff| > max range):")
    for i, a in enumerate(names):
        for bname in names[i + 1:]:
            sig, diff = differs(s[a], s[bname])
            bound = max(s[a]["M2"]["range"], s[bname]["M2"]["range"])
            print(f"  {a} - {bname}: {diff:+.6f} vs {bound:.6f} -> {'差あり' if sig else '差なし'}")
    kind, holds = result_type(s)
    print(f"result type: {kind} (conditions holding: {holds or 'none'})")
    chosen, detail = candidate(s)
    for name, cut in detail["cuts"].items():
        print(f"  cut {name}: " + ", ".join(f"{k} {'ok' if v else 'NG'}" for k, v in cut.items()))
    print(f"passing: {detail['passing']}; best M2: {detail.get('best_M2')}; "
          f"tied with best: {detail.get('tied_with_best')}")
    print(f"final candidate: {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
