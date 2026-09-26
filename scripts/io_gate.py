"""Apply the rules pre-registered in docs/input_order.md §4, mechanically.

    uv run python scripts/io_gate.py --new runs/io/eval_seed0.json          # breakage
    uv run python scripts/io_gate.py --new runs/io/eval_seed{0,1,2}.json    # verdicts

With one new checkpoint it checks only the breakage stop. With three it prints the
hypothesis verdict (supported / rejected / partial) and the adoption-candidate verdict,
each with the numbers that decided it. Compares the double-precision values the
evaluation wrote; nothing is rounded first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GUARDS = [("choice_accuracy", "higher"), ("score_rps", "lower"),
          ("score_accuracy", "higher"), ("bool_accuracy", "higher")]
M4_MIN_N = 30


def metric(entry: dict, name: str) -> float:
    if name == "M1":
        return entry["M1"]["auroc"]
    if name == "M2":
        return entry["M2"]["auroc"]
    if name == "M3":
        return entry["M3"]["accuracy"]
    if name == "M4":
        return entry["M4"]["accuracy"]
    return entry["guardrails"][name]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="runs/io/eval_v01.json")
    parser.add_argument("--new", nargs="+", required=True)
    args = parser.parse_args()

    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    new = [e for p in args.new for e in json.loads(Path(p).read_text(encoding="utf-8"))]
    assert len(base) == 3 and all(e["input_order"] == "question_first" for e in base)
    assert all(e["input_order"] == "state_first" and e["local_attention"] == 128
               and not e["inference_only_override"] for e in new)

    def vals(rows: list[dict], name: str) -> list[float]:
        return [metric(e, name) for e in rows]

    def mean(rows: list[dict], name: str) -> float:
        v = vals(rows, name)
        return sum(v) / len(v)

    b_m1 = vals(base, "M1")
    floor = min(b_m1) - 2 * (max(b_m1) - min(b_m1))
    seed0 = new[0]
    broken = metric(seed0, "M1") < floor
    print(f"breakage: seed 0 M1 {metric(seed0, 'M1'):.6f} vs floor {floor:.6f} "
          f"(= {min(b_m1):.6f} - 2 x {max(b_m1) - min(b_m1):.6f}) -> "
          f"{'BROKEN, stop' if broken else 'not broken'}")
    if broken or len(new) == 1:
        return 0

    m3_mean, m3_max = mean(new, "M3"), max(vals(base, "M3"))
    m4_usable = all(e["M4"]["n"] >= M4_MIN_N for e in new + base)
    m4_mean, m4_min = mean(new, "M4"), min(vals(base, "M4"))
    print(f"M3 mean {m3_mean:.6f} vs v0.1 max {m3_max:.6f} -> "
          f"{'above' if m3_mean > m3_max else 'not above'}")
    print(f"M4 mean {m4_mean:.6f} vs v0.1 min {m4_min:.6f} -> "
          f"{'below' if m4_mean < m4_min else 'not below'}"
          f"{'' if m4_usable else ' (M4 unusable: n < 30)'}")
    if m3_mean <= m3_max:
        hypothesis = "REJECTED"
    elif not m4_usable or m4_mean < m4_min:
        hypothesis = "SUPPORTED"
    else:
        hypothesis = "PARTIAL"
    print(f"distance hypothesis: {hypothesis}")

    improve = []
    for name in ("M2", "M3"):
        ok = mean(new, name) > max(vals(base, name))
        improve.append(ok)
        print(f"  {name} mean {mean(new, name):.6f} vs v0.1 max {max(vals(base, name)):.6f}"
              f" -> {'above' if ok else 'not above'}")
    checks = [any(improve), mean(new, "M1") >= min(b_m1)]
    print(f"  M1 mean {mean(new, 'M1'):.6f} vs v0.1 min {min(b_m1):.6f} -> "
          f"{'ok' if checks[-1] else 'below'}")
    for name, direction in GUARDS:
        b = vals(base, name)
        worst = max(b) if direction == "lower" else min(b)
        value = mean(new, name)
        ok = value <= worst if direction == "lower" else value >= worst
        checks.append(ok)
        print(f"  {name} mean {value:.6f} vs v0.1 worst {worst:.6f} -> "
              f"{'ok' if ok else 'BROKEN'}")
    print(f"adoption candidate: {'YES' if all(checks) else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
