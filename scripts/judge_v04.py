"""Night 3 §3: place v0.4 against the three-seed ranges of v0.1 and v0.3.

    uv run python scripts/judge_v04.py

Two yardsticks, both already scored for v0.1 and v0.3 on three seeds each:

* the frozen one, `data/v2/heldout_val_v2.jsonl` -- five attributes, the set v0.1 was
  judged on (`runs/seedvar_<run>.json`);
* the twelve-attribute held-out, `data/v4/heldout_val_v4.jsonl`, grouped into the five
  v0.1 attributes and the seven added on Day 3 (`runs/diagnostics/heldout_compare_*.json`).

A run is INSIDE a range if it falls between that range's lowest and highest seed. With
three seeds that range is narrow and one v0.4 seed is one draw, so ABOVE is necessary
evidence of a gain, not sufficient -- the report says which it is.

H2's direct test is the tier-I pool on the seven Day-3 attributes: none of v0.1, v0.3
or v0.4 trained on them, so a v0.4 above v0.1's range there is transfer that the
re-admitted attributes bought.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

FROZEN = ("overall", "E", "I", "S")
GROUPS = ("Day 3 で追加/I", "Day 3 で追加/E", "Day 3 で追加/S", "Day 3 で追加/all",
          "v0.1 の 5 属性/I", "v0.1 の 5 属性/all", "全体/I", "全体/all")
DIRECT = "Day 3 で追加/I"


def frozen_value(run: dict[str, Any], key: str) -> float:
    return run["overall"]["auroc"] if key == "overall" else run["by_tier"][key]["auroc"]


def place(value: float, values: list[float]) -> str:
    lo, hi = min(values), max(values)
    return "INSIDE" if lo <= value <= hi else ("ABOVE" if value > hi else "BELOW")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="v04_seed0")
    parser.add_argument("--compare-3x3", default="runs/diagnostics/heldout_compare_3x3.json")
    parser.add_argument("--compare-new", default="runs/diagnostics/heldout_compare_v04.json")
    parser.add_argument("--out", default="runs/diagnostics/judge_v04.json")
    args = parser.parse_args()

    def seedvar(name: str) -> dict[str, Any]:
        return json.loads(Path(f"runs/seedvar_{name}.json").read_text(encoding="utf-8"))

    refs = {"v0.1": [seedvar(f"v01_seed{i}") for i in range(3)],
            "v0.3": [seedvar(f"v03_seed{i}") for i in range(3)]}
    new = seedvar(args.run)
    result: dict[str, Any] = {"run": args.run, "frozen": {}, "twelve": {}}

    print(f"== 凍結物差し heldout_val_v2（5 属性、9,850 ビュー）: {args.run} ==")
    print(f"{'metric':10s}{'v0.1 range':>20s}{'v0.3 range':>20s}{args.run:>12s}"
          f"{'vs v0.1':>9s}{'vs v0.3':>9s}")
    for key in FROZEN:
        v = frozen_value(new, key)
        row = {"value": v}
        cells = []
        for ref, runs in refs.items():
            vals = [frozen_value(r, key) for r in runs]
            row[ref] = {"min": min(vals), "max": max(vals), "mean": st.mean(vals),
                        "place": place(v, vals)}
            cells.append(f"[{min(vals):.4f},{max(vals):.4f}]")
        result["frozen"][key] = row
        print(f"{key:10s}{cells[0]:>20s}{cells[1]:>20s}{v:12.4f}"
              f"{row['v0.1']['place']:>9s}{row['v0.3']['place']:>9s}")

    old = json.loads(Path(args.compare_3x3).read_text(encoding="utf-8"))["by_checkpoint"]
    cur = json.loads(Path(args.compare_new).read_text(encoding="utf-8"))["by_checkpoint"]
    (checkpoint, scores), = [(k, v) for k, v in cur.items() if args.run in k]
    ref12 = {"v0.1": [old[f"runs/v01_seed{i}/model.pt"] for i in range(3)],
             "v0.3": [old[f"runs/v03_seed{i}/model.pt"] for i in range(3)]}

    print(f"\n== 12 属性 held-out heldout_val_v4（23,910 ビュー）: {checkpoint} ==")
    print(f"{'group':20s}{'v0.1 range':>20s}{'v0.3 range':>20s}{args.run:>12s}"
          f"{'vs v0.1':>9s}{'vs v0.3':>9s}")
    for key in GROUPS:
        v = scores[key]["auroc"]
        row = {"value": v, "n": scores[key]["n"]}
        cells = []
        for ref, runs in ref12.items():
            vals = [r[key]["auroc"] for r in runs]
            row[ref] = {"min": min(vals), "max": max(vals), "mean": st.mean(vals),
                        "sd": st.pstdev(vals), "place": place(v, vals)}
            cells.append(f"[{min(vals):.4f},{max(vals):.4f}]")
        result["twelve"][key] = row
        mark = "  <- H2" if key == DIRECT else ""
        print(f"{key:20s}{cells[0]:>20s}{cells[1]:>20s}{v:12.4f}"
              f"{row['v0.1']['place']:>9s}{row['v0.3']['place']:>9s}{mark}")

    direct = result["twelve"][DIRECT]
    above = direct["v0.1"]["place"] == "ABOVE"
    margin = direct["value"] - direct["v0.1"]["max"]
    sd = direct["v0.1"]["sd"]
    result["h2_direct"] = {
        "value": direct["value"], "v0.1_max": direct["v0.1"]["max"],
        "v0.1_mean": direct["v0.1"]["mean"], "v0.1_sd": sd,
        "above_v0.1_range": above, "margin_over_max": margin,
    }
    print(f"\nH2 の直接の検証（{DIRECT}）: {direct['value']:.4f} vs v0.1 "
          f"[{direct['v0.1']['min']:.4f}, {direct['v0.1']['max']:.4f}] "
          f"mean {direct['v0.1']['mean']:.4f} sd {sd:.4f} -> {direct['v0.1']['place']}"
          f"  (最大値との差 {margin:+.4f}, v0.1 平均との差 "
          f"{direct['value'] - direct['v0.1']['mean']:+.4f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
