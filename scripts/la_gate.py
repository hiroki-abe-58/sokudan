"""Apply the gates pre-registered in docs/local_attention_1024.md §3-4, mechanically.

    uv run python scripts/la_gate.py --baseline runs/la/eval_v01.json \
        --new runs/la/eval_la1024.json

One new checkpoint -> the advance gate (new seed 0 against v0.1 seed 0 and the v0.1
3-seed spread). Three -> the adoption gate (new means against v0.1 min / max). Reads
the double-precision values the evaluation wrote; nothing is rounded before comparing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GUARDS = [("choice_accuracy", "higher"), ("score_rps", "lower"),
          ("score_accuracy", "higher"), ("bool_accuracy", "higher")]


def metric(entry: dict, name: str) -> float:
    if name == "M1":
        return entry["M1"]["auroc"]
    if name == "M2":
        return entry["M2"]["auroc"]
    if name == "M3":
        return entry["M3"]["accuracy"]
    return entry["guardrails"][name]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="runs/la/eval_v01.json")
    parser.add_argument("--new", nargs="+", required=True)
    args = parser.parse_args()

    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    new = [e for path in args.new for e in json.loads(Path(path).read_text(encoding="utf-8"))]
    assert len(base) == 3 and all(e["local_attention"] == 128 for e in base)
    assert all(e["local_attention"] == 1024 and not e["inference_only_override"]
               for e in new)

    def values(rows: list[dict], name: str) -> list[float]:
        return [metric(e, name) for e in rows]

    lines, checks = [], []
    if len(new) == 1:
        kind = "advance (new seed 0)"
        n = new[0]
        improve = []
        for name in ("M2", "M3"):
            b = values(base, name)
            spread = max(b) - min(b)
            delta = metric(n, name) - b[0]
            ok = delta > spread
            improve.append(ok)
            lines.append(f"  {name}: new {metric(n, name):.6f} - v0.1 s0 {b[0]:.6f} = "
                         f"{delta:+.6f} vs spread {spread:.6f} -> {'yes' if ok else 'no'}")
        checks.append(("M2 or M3 improvement", any(improve)))
        m1 = min(values(base, "M1"))
        checks.append(("M1 >= v0.1 min", metric(n, "M1") >= m1))
        lines.append(f"  M1: new {metric(n, 'M1'):.6f} vs v0.1 min {m1:.6f}")
        for name, direction in GUARDS:
            b = values(base, name)
            worst = max(b) if direction == "lower" else min(b)
            ok = metric(n, name) <= worst if direction == "lower" else metric(n, name) >= worst
            checks.append((f"{name} vs v0.1 worst", ok))
            lines.append(f"  {name}: new {metric(n, name):.6f} vs v0.1 worst {worst:.6f} "
                         f"({direction} is better) -> {'ok' if ok else 'BROKEN'}")
    else:
        kind = f"adoption ({len(new)} seeds)"

        def mean(rows: list[dict], name: str) -> float:
            v = values(rows, name)
            return sum(v) / len(v)

        improve = []
        for name in ("M2", "M3"):
            ok = mean(new, name) > max(values(base, name))
            improve.append(ok)
            lines.append(f"  {name}: new mean {mean(new, name):.6f} vs v0.1 max "
                         f"{max(values(base, name)):.6f} -> {'yes' if ok else 'no'}")
        checks.append(("M2 or M3 mean > v0.1 max", any(improve)))
        m1 = min(values(base, "M1"))
        checks.append(("M1 mean >= v0.1 min", mean(new, "M1") >= m1))
        lines.append(f"  M1: new mean {mean(new, 'M1'):.6f} vs v0.1 min {m1:.6f}")
        for name, direction in GUARDS:
            b = values(base, name)
            worst = max(b) if direction == "lower" else min(b)
            value = mean(new, name)
            ok = value <= worst if direction == "lower" else value >= worst
            checks.append((f"{name} mean vs v0.1 worst", ok))
            lines.append(f"  {name}: new mean {value:.6f} vs v0.1 worst {worst:.6f} "
                         f"-> {'ok' if ok else 'BROKEN'}")

    verdict = "PASS" if all(ok for _, ok in checks) else "FAIL"
    print(f"gate: {kind}")
    print("\n".join(lines))
    for name, ok in checks:
        print(f"  [{'x' if ok else ' '}] {name}")
    print(f"{verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
